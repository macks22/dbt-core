from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from dbt.clients.jinja import MacroGenerator, MacroStack
from dbt.contracts.graph.nodes import Macro
from dbt.exceptions import DuplicateMacroNameError, PackageNotFoundForMacroError
from dbt.include.global_project import PROJECT_NAME as GLOBAL_PROJECT_NAME

FlatNamespace = Dict[str, MacroGenerator]
NamespaceMember = Union[FlatNamespace, MacroGenerator]
FullNamespace = Dict[str, NamespaceMember]

# Wider value type for the lazy namespace; covers the case where __getitem__
# returns a _LazyPackageNamespace (a Mapping[str, MacroGenerator] view) instead
# of a concrete dict. Behaves identically from jinja's perspective.
LazyNamespaceMember = Union[Mapping[str, MacroGenerator], MacroGenerator]


@dataclass(frozen=True)
class NamespaceTemplateKey:
    root_package: str
    search_package: str
    internal_packages: Tuple[str, ...]


@dataclass
class NamespaceTemplate:
    """Macro routing computed once per (root, search, internal_packages) combo.

    Holds bare ``Macro`` references (no ``MacroGenerator`` wrappers), so the
    template is independent of any per-node ``ctx``/``node``/``stack`` and can
    be cached on the ``Manifest``.
    """

    globals_macros: Dict[str, Macro] = field(default_factory=dict)
    locals_macros: Dict[str, Macro] = field(default_factory=dict)
    packages_macros: Dict[str, Dict[str, Macro]] = field(default_factory=dict)
    internal_packages_macros: Dict[str, Dict[str, Macro]] = field(default_factory=dict)
    global_project_macros: Dict[str, Macro] = field(default_factory=dict)


# The point of this class is to collect the various macros
# and provide the ability to flatten them into the ManifestContexts
# that are created for jinja, so that macro calls can be resolved.
# Creates special iterators and _keys methods to flatten the lists.
# When this class is created it has a static 'local_namespace' which
# depends on the package of the node, so it only works for one
# particular local package at a time for "flattening" into a context.
# 'get_by_package' should work for any macro.
class MacroNamespace(Mapping):
    def __init__(
        self,
        global_namespace: FlatNamespace,  # root package macros
        local_namespace: FlatNamespace,  # packages for *this* node
        global_project_namespace: FlatNamespace,  # internal packages
        packages: Dict[str, FlatNamespace],  # non-internal packages
    ):
        self.global_namespace: FlatNamespace = global_namespace
        self.local_namespace: FlatNamespace = local_namespace
        self.packages: Dict[str, FlatNamespace] = packages
        self.global_project_namespace: FlatNamespace = global_project_namespace

    def _search_order(self) -> Iterable[Union[FullNamespace, FlatNamespace]]:
        yield self.local_namespace  # local package
        yield self.global_namespace  # root package
        # TODO CT-211
        yield self.packages  # type: ignore[misc] # non-internal packages
        yield {
            # TODO CT-211
            GLOBAL_PROJECT_NAME: self.global_project_namespace,  # type: ignore[misc] # dbt
        }
        yield self.global_project_namespace  # other internal project besides dbt

    # provides special keys method for MacroNamespace iterator
    # returns keys from local_namespace, global_namespace, packages,
    # global_project_namespace
    def _keys(self) -> Set[str]:
        keys: Set[str] = set()
        for search in self._search_order():
            keys.update(search)
        return keys

    # special iterator using special keys
    def __iter__(self) -> Iterator[str]:
        for key in self._keys():
            yield key

    def __len__(self):
        return len(self._keys())

    def __getitem__(self, key: str) -> NamespaceMember:
        for dct in self._search_order():
            if key in dct:
                return dct[key]
        raise KeyError(key)

    def get_from_package(self, package_name: Optional[str], name: str) -> Optional[MacroGenerator]:
        if package_name is None:
            return self.get(name)
        elif package_name == GLOBAL_PROJECT_NAME:
            return self.global_project_namespace.get(name)
        elif package_name in self.packages:
            return self.packages[package_name].get(name)
        else:
            raise PackageNotFoundForMacroError(package_name)


# This class builds the MacroNamespace by adding macros to
# internal_packages or packages, and locals/globals.
# Call 'build_namespace' to return a MacroNamespace.
# This is used by ManifestContext (and subclasses)
class MacroNamespaceBuilder:
    def __init__(
        self,
        root_package: str,
        search_package: str,
        thread_ctx: MacroStack,
        internal_packages: List[str],
        node: Optional[Any] = None,
    ) -> None:
        self.root_package = root_package
        self.search_package = search_package
        # internal packages comes from get_adapter_package_names
        self.internal_package_names = set(internal_packages)
        self.internal_package_names_order = internal_packages
        # macro_func is added here if in root package, since
        # the root package acts as a "global" namespace, overriding
        # everything else except local external package macro calls
        self.globals: FlatNamespace = {}
        # macro_func is added here if it's the package for this node
        self.locals: FlatNamespace = {}
        # Create a dictionary of [package name][macro name] =
        #     MacroGenerator object which acts like a function
        self.internal_packages: Dict[str, FlatNamespace] = {}
        self.packages: Dict[str, FlatNamespace] = {}
        self.thread_ctx = thread_ctx
        self.node = node

    def _add_macro_to(
        self,
        hierarchy: Dict[str, FlatNamespace],
        macro: Macro,
        macro_func: MacroGenerator,
    ):
        if macro.package_name in hierarchy:
            namespace = hierarchy[macro.package_name]
        else:
            namespace = {}
            hierarchy[macro.package_name] = namespace

        if macro.name in namespace:
            raise DuplicateMacroNameError(macro_func.macro, macro, macro.package_name)
        hierarchy[macro.package_name][macro.name] = macro_func

    def add_macro(self, macro: Macro, ctx: Dict[str, Any]) -> None:
        macro_name: str = macro.name

        # MacroGenerator is in clients/jinja.py
        # a MacroGenerator object is a callable object that will
        # execute the MacroGenerator.__call__ function
        macro_func: MacroGenerator = MacroGenerator(macro, ctx, self.node, self.thread_ctx)

        # internal macros (from plugins) will be processed separately from
        # project macros, so store them in a different place
        if macro.package_name in self.internal_package_names:
            self._add_macro_to(self.internal_packages, macro, macro_func)
        else:
            # if it's not an internal package
            self._add_macro_to(self.packages, macro, macro_func)
            # add to locals if it's the package this node is in
            if macro.package_name == self.search_package:
                self.locals[macro_name] = macro_func
            # add to globals if it's in the root package
            elif macro.package_name == self.root_package:
                self.globals[macro_name] = macro_func

    def add_macros(self, macros: Iterable[Macro], ctx: Dict[str, Any]) -> None:
        for macro in macros:
            self.add_macro(macro, ctx)

    def build_namespace(
        self, macros_by_package: Dict[str, Dict[str, Macro]], ctx: Dict[str, Any]
    ) -> MacroNamespace:
        for package in macros_by_package.values():
            self.add_macros(package.values(), ctx)

        # Iterate in reverse-order and overwrite: the packages that are first
        # in the list are the ones we want to "win".
        global_project_namespace: FlatNamespace = {}
        for pkg in reversed(self.internal_package_names_order):
            if pkg in self.internal_packages:
                # add the macros pointed to by this package name
                global_project_namespace.update(self.internal_packages[pkg])

        return MacroNamespace(
            global_namespace=self.globals,  # root package macros
            local_namespace=self.locals,  # packages for *this* node
            global_project_namespace=global_project_namespace,  # internal packages
            packages=self.packages,  # non internal_packages
        )

    @classmethod
    def build_template(
        cls,
        root_package: str,
        search_package: str,
        internal_packages: List[str],
        macros_by_package: Dict[str, Dict[str, Macro]],
    ) -> NamespaceTemplate:
        """Build a per-(root, search, internal_packages) routing of bare macros.

        This is the slow part of namespace construction (one pass over every
        macro in the project), but the result has no per-node state, so the
        caller can cache it on the ``Manifest`` and reuse for every node that
        shares the same key.
        """
        template = NamespaceTemplate()
        internal_set = set(internal_packages)
        for package in macros_by_package.values():
            for macro in package.values():
                if macro.package_name in internal_set:
                    bucket = template.internal_packages_macros.setdefault(macro.package_name, {})
                    if macro.name in bucket:
                        raise DuplicateMacroNameError(macro, macro, macro.package_name)
                    bucket[macro.name] = macro
                else:
                    bucket = template.packages_macros.setdefault(macro.package_name, {})
                    if macro.name in bucket:
                        raise DuplicateMacroNameError(macro, macro, macro.package_name)
                    bucket[macro.name] = macro
                    if macro.package_name == search_package:
                        template.locals_macros[macro.name] = macro
                    elif macro.package_name == root_package:
                        template.globals_macros[macro.name] = macro

        # Flatten internal packages in reverse order so the first-listed
        # internal package wins, matching MacroNamespaceBuilder.build_namespace.
        for pkg in reversed(internal_packages):
            if pkg in template.internal_packages_macros:
                template.global_project_macros.update(template.internal_packages_macros[pkg])
        return template


class _LazyPackageNamespace(Mapping):
    """Mapping view over one package's macros that wraps on access.

    Used when ``LazyMacroNamespace[<package_name>]`` returns a nested mapping
    from macro name to ``MacroGenerator``. Wrappers come from the parent's
    cache so each (template, ctx, node) combination wraps each macro at most
    once.
    """

    def __init__(self, parent: "LazyMacroNamespace", package: str, is_internal: bool):
        self._parent = parent
        self._package = package
        self._is_internal = is_internal

    def _macros(self) -> Dict[str, Macro]:
        if self._is_internal:
            return self._parent._template.global_project_macros
        return self._parent._template.packages_macros.get(self._package, {})

    def __getitem__(self, key: str) -> MacroGenerator:
        macros = self._macros()
        if key not in macros:
            raise KeyError(key)
        return self._parent._wrap(macros[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._macros())

    def __len__(self) -> int:
        return len(self._macros())


class LazyMacroNamespace(Mapping):
    """Drop-in replacement for ``MacroNamespace`` that wraps macros lazily.

    Same external API as ``MacroNamespace``: ``__getitem__``, ``__iter__``,
    ``__len__``, ``get_from_package``, and the four flat-namespace properties
    (``global_namespace``, ``local_namespace``, ``packages``,
    ``global_project_namespace``). Wrappers are created on first access and
    cached per macro ``unique_id``, so repeated lookups within one parse-node
    pay only the first wrap cost.
    """

    def __init__(
        self,
        template: NamespaceTemplate,
        ctx: Dict[str, Any],
        node: Optional[Any],
        stack: MacroStack,
    ):
        self._template = template
        self._ctx = ctx
        self._node = node
        self._stack = stack
        self._wrap_cache: Dict[str, MacroGenerator] = {}

    def _wrap(self, macro: Macro) -> MacroGenerator:
        cached = self._wrap_cache.get(macro.unique_id)
        if cached is not None:
            return cached
        wrapper = MacroGenerator(macro, self._ctx, self._node, self._stack)
        self._wrap_cache[macro.unique_id] = wrapper
        return wrapper

    def set_ctx(self, ctx: Dict[str, Any]) -> None:
        """Swap the ctx that future MacroGenerator wrappers will close over.

        Called by ``ManifestContext.to_dict`` after the ``MacroDictProxy`` is
        built so macro bodies can resolve sibling macros (e.g. ``pkg.macro``)
        through the proxy. The wrap cache is cleared because any wrapper
        created before the swap captured the bare base ctx.
        """
        self._ctx = ctx
        self._wrap_cache.clear()

    @property
    def global_namespace(self) -> FlatNamespace:
        return {n: self._wrap(m) for n, m in self._template.globals_macros.items()}

    @property
    def local_namespace(self) -> FlatNamespace:
        return {n: self._wrap(m) for n, m in self._template.locals_macros.items()}

    @property
    def packages(self) -> Dict[str, FlatNamespace]:
        return {
            p: {n: self._wrap(m) for n, m in ms.items()}
            for p, ms in self._template.packages_macros.items()
        }

    @property
    def global_project_namespace(self) -> FlatNamespace:
        return {n: self._wrap(m) for n, m in self._template.global_project_macros.items()}

    def __getitem__(self, key: str) -> LazyNamespaceMember:
        # Same precedence as MacroNamespace._search_order: locals, globals,
        # packages (nested dict), {GLOBAL_PROJECT_NAME: internal}, global_project.
        if key in self._template.locals_macros:
            return self._wrap(self._template.locals_macros[key])
        if key in self._template.globals_macros:
            return self._wrap(self._template.globals_macros[key])
        if key in self._template.packages_macros:
            return _LazyPackageNamespace(self, key, is_internal=False)
        if key == GLOBAL_PROJECT_NAME:
            return _LazyPackageNamespace(self, key, is_internal=True)
        if key in self._template.global_project_macros:
            return self._wrap(self._template.global_project_macros[key])
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        keys: Set[str] = set()
        keys.update(self._template.locals_macros)
        keys.update(self._template.globals_macros)
        keys.update(self._template.packages_macros)
        keys.add(GLOBAL_PROJECT_NAME)
        keys.update(self._template.global_project_macros)
        return iter(keys)

    def __len__(self) -> int:
        keys: Set[str] = set()
        keys.update(self._template.locals_macros)
        keys.update(self._template.globals_macros)
        keys.update(self._template.packages_macros)
        keys.add(GLOBAL_PROJECT_NAME)
        keys.update(self._template.global_project_macros)
        return len(keys)

    def get_from_package(self, package_name: Optional[str], name: str) -> Optional[MacroGenerator]:
        if package_name is None:
            value = self.get(name)
            return value if isinstance(value, MacroGenerator) else None
        if package_name == GLOBAL_PROJECT_NAME:
            macro = self._template.global_project_macros.get(name)
            return self._wrap(macro) if macro is not None else None
        package = self._template.packages_macros.get(package_name)
        if package is None:
            raise PackageNotFoundForMacroError(package_name)
        macro = package.get(name)
        return self._wrap(macro) if macro is not None else None
