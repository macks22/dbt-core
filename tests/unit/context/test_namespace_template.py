import unittest
from unittest import mock

from dbt.clients.jinja import MacroGenerator, MacroStack
from dbt.context.macros import (
    LazyMacroNamespace,
    MacroNamespaceBuilder,
    NamespaceTemplate,
    _LazyPackageNamespace,
)
from dbt.contracts.graph.nodes import Macro
from dbt.exceptions import DuplicateMacroNameError, PackageNotFoundForMacroError
from dbt.include.global_project import PROJECT_NAME as GLOBAL_PROJECT_NAME


def make_macro(name: str, package: str) -> Macro:
    macro = mock.MagicMock(
        __class__=Macro,
        package_name=package,
        resource_type="macro",
        unique_id=f"macro.{package}.{name}",
    )
    macro.name = name
    return macro


def macros_by_package(*macros):
    by_package: dict = {}
    for m in macros:
        by_package.setdefault(m.package_name, {})[m.name] = m
    return by_package


class TestBuildTemplate(unittest.TestCase):
    """``MacroNamespaceBuilder.build_template`` produces routing without ctx/node/stack."""

    def test_routes_root_search_internal(self):
        # search package contributes locals; root contributes globals; an
        # internal package fills global_project; everything goes in packages or
        # internal_packages_macros.
        m_search = make_macro("local_macro", "search_pkg")
        m_root = make_macro("global_macro", "root_pkg")
        m_pkg = make_macro("third_party_macro", "third_party")
        m_internal = make_macro("internal_macro", "dbt")

        template = MacroNamespaceBuilder.build_template(
            root_package="root_pkg",
            search_package="search_pkg",
            internal_packages=["dbt"],
            macros_by_package=macros_by_package(m_search, m_root, m_pkg, m_internal),
        )

        self.assertIs(template.locals_macros["local_macro"], m_search)
        self.assertIs(template.globals_macros["global_macro"], m_root)
        self.assertIs(template.packages_macros["third_party"]["third_party_macro"], m_pkg)
        self.assertIs(template.internal_packages_macros["dbt"]["internal_macro"], m_internal)
        self.assertIs(template.global_project_macros["internal_macro"], m_internal)
        # Internal macros must not leak into the non-internal packages bucket.
        self.assertNotIn("dbt", template.packages_macros)

    def test_internal_first_listed_wins(self):
        # When two internal packages define a macro with the same name, the
        # first package in the order list takes precedence in
        # global_project_macros (matches build_namespace).
        m_a = make_macro("shared", "pkg_a")
        m_b = make_macro("shared", "pkg_b")

        template = MacroNamespaceBuilder.build_template(
            root_package="root_pkg",
            search_package="search_pkg",
            internal_packages=["pkg_a", "pkg_b"],
            macros_by_package=macros_by_package(m_a, m_b),
        )

        self.assertIs(template.global_project_macros["shared"], m_a)

    def test_duplicate_in_same_package_raises(self):
        # Two macros with the same name in the same package is a real
        # collision and must surface immediately.
        m1 = make_macro("dupe", "pkg")
        m2 = make_macro("dupe", "pkg")
        with self.assertRaises(DuplicateMacroNameError):
            MacroNamespaceBuilder.build_template(
                root_package="pkg",
                search_package="pkg",
                internal_packages=[],
                macros_by_package={"pkg": {"dupe_a": m1, "dupe_b": m2}},
            )


class TestLazyMacroNamespace(unittest.TestCase):
    """``LazyMacroNamespace`` wraps macros only on access and caches by unique_id."""

    def _build(self, search_package="search_pkg", root_package="root_pkg", internals=("dbt",)):
        macros = [
            make_macro("local_macro", search_package),
            make_macro("global_macro", root_package),
            make_macro("third_party_macro", "third_party"),
            make_macro("internal_macro", "dbt"),
        ]
        template = MacroNamespaceBuilder.build_template(
            root_package=root_package,
            search_package=search_package,
            internal_packages=list(internals),
            macros_by_package=macros_by_package(*macros),
        )
        ns = LazyMacroNamespace(template=template, ctx={}, node=None, stack=MacroStack())
        return ns, macros

    def test_lookup_precedence_matches_eager_namespace(self):
        # Search order: locals, globals, packages (nested), GLOBAL_PROJECT_NAME,
        # global_project_macros (flat).
        ns, _ = self._build()

        self.assertIsInstance(ns["local_macro"], MacroGenerator)
        self.assertIsInstance(ns["global_macro"], MacroGenerator)
        self.assertIsInstance(ns["third_party"], _LazyPackageNamespace)
        self.assertIsInstance(ns[GLOBAL_PROJECT_NAME], _LazyPackageNamespace)
        self.assertIsInstance(ns["internal_macro"], MacroGenerator)

        with self.assertRaises(KeyError):
            ns["does_not_exist"]

    def test_wrap_cache_returns_same_generator(self):
        # Repeated access wraps once — the perf invariant.
        ns, _ = self._build()
        first = ns["local_macro"]
        second = ns["local_macro"]
        self.assertIs(first, second)

    def test_set_ctx_clears_cache(self):
        # set_ctx must invalidate previously-cached wrappers because they
        # closed over the old ctx.
        ns, _ = self._build()
        first = ns["local_macro"]
        ns.set_ctx({"new": "ctx"})
        second = ns["local_macro"]
        self.assertIsNot(first, second)
        self.assertIs(second.context, ns._ctx)

    def test_get_from_package_paths(self):
        ns, _ = self._build()
        # None → flat search via __getitem__
        gen = ns.get_from_package(None, "local_macro")
        self.assertIsInstance(gen, MacroGenerator)
        # internal package via GLOBAL_PROJECT_NAME alias
        gen = ns.get_from_package(GLOBAL_PROJECT_NAME, "internal_macro")
        self.assertIsInstance(gen, MacroGenerator)
        # named non-internal package
        gen = ns.get_from_package("third_party", "third_party_macro")
        self.assertIsInstance(gen, MacroGenerator)
        # Missing macro in a known package returns None (not raise)
        self.assertIsNone(ns.get_from_package("third_party", "missing"))
        # Unknown package raises (matches MacroNamespace.get_from_package)
        with self.assertRaises(PackageNotFoundForMacroError):
            ns.get_from_package("nonexistent_package", "x")

    def test_iter_and_len_cover_all_buckets(self):
        ns, _ = self._build()
        keys = set(ns)
        self.assertIn("local_macro", keys)
        self.assertIn("global_macro", keys)
        self.assertIn("third_party", keys)
        self.assertIn(GLOBAL_PROJECT_NAME, keys)
        self.assertIn("internal_macro", keys)
        self.assertEqual(len(ns), len(keys))


class TestNamespaceTemplateDefaults(unittest.TestCase):
    """Bare ``NamespaceTemplate`` should be usable with empty buckets."""

    def test_empty_template_lookup_raises_keyerror(self):
        ns = LazyMacroNamespace(
            template=NamespaceTemplate(),
            ctx={},
            node=None,
            stack=MacroStack(),
        )
        with self.assertRaises(KeyError):
            ns["anything"]
