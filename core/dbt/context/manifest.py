from typing import Any, List, Union

from dbt.adapters.contracts.connection import AdapterRequiredConfig
from dbt.clients.jinja import MacroStack
from dbt.context.macro_resolver import TestMacroNamespace
from dbt.contracts.graph.manifest import Manifest

from .base import contextproperty
from .configured import ConfiguredContext
from .macros import LazyMacroNamespace, MacroNamespace, MacroNamespaceBuilder


class MacroDictProxy(dict):
    """Dict that materializes macro names on demand from a LazyMacroNamespace.

    Jinja accesses ``ctx[name]`` when a macro is invoked, so the only macros we
    pay to wrap are the ones the rendered template actually references.
    ``__missing__`` mutates self with the resolved value on first access.

    ``__contains__`` and ``get`` must also consult the namespace because
    Python's ``dict`` only calls ``__missing__`` from ``__getitem__`` — Jinja's
    ``Context.resolve`` uses ``in`` before subscripting, and other call sites
    (e.g. unit-test override resolution) use ``in``/``get`` directly.
    """

    __slots__ = ("_ns",)

    def __init__(self, base: dict, ns: LazyMacroNamespace):
        super().__init__(base)
        self._ns = ns
        # Macros use ``{{ context[macro_name](...) }}`` for dynamic dispatch.
        # ``base`` already has ``context`` pointing at the bare ctx (set by
        # ``BaseContext.to_dict``); rewire it to the proxy so dynamic lookups
        # see the lazy namespace.
        self["context"] = self

    def materialize_builtin_overrides(self) -> None:
        """Eagerly materialize macros that shadow builtins in the dict.

        The eager ``MacroNamespace`` path achieves override-of-builtin via
        ``dct.update(self.namespace)``: every macro key is written into the
        dict, so lookups land on the override. In the lazy path,
        ``dict.__getitem__`` finds the builtin first and never consults
        ``__missing__``, so the override would be silently lost.

        Caller MUST call ``ns.set_ctx(self)`` first so the materialized
        wrappers close over this proxy (not the bare ``_ctx``).
        """
        ns = self._ns
        for key in ns:
            if dict.__contains__(self, key):
                self[key] = ns[key]

    def __missing__(self, key: str) -> Any:
        try:
            value = self._ns[key]
        except KeyError:
            raise KeyError(key)
        self[key] = value
        return value

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        return isinstance(key, str) and key in self._ns

    def get(self, key, default=None):
        if super().__contains__(key):
            return super().__getitem__(key)
        try:
            return self.__getitem__(key)
        except KeyError:
            return default


class ManifestContext(ConfiguredContext):
    """The Macro context has everything in the target context, plus the macros
    in the manifest.

    The given macros can override any previous context values, which will be
    available as if they were accessed relative to the package name.
    """

    # subclasses are QueryHeaderContext and ProviderContext
    def __init__(
        self,
        config: AdapterRequiredConfig,
        manifest: Manifest,
        search_package: str,
    ) -> None:
        super().__init__(config)
        self.manifest = manifest
        # this is the package of the node for which this context was built
        self.search_package = search_package
        self.macro_stack = MacroStack()
        # This namespace is used by the BaseDatabaseWrapper in jinja rendering.
        # The namespace is passed to it when it's constructed. It expects
        # to be able to do: namespace.get_from_package(..)
        self.namespace = self._build_namespace()

    def _build_namespace(self) -> Union[MacroNamespace, LazyMacroNamespace]:
        # avoid an import loop
        from dbt.adapters.factory import get_adapter_package_names

        internal_packages: List[str] = get_adapter_package_names(self.config.credentials.type)
        template = self.manifest.get_namespace_template(
            root_package=self.config.project_name,
            search_package=self.search_package,
            internal_packages=tuple(internal_packages),
        )
        return LazyMacroNamespace(
            template=template,
            ctx=self._ctx,
            node=self._namespace_node(),
            stack=self.macro_stack,
        )

    def _namespace_node(self):
        """Return the node MacroGenerators should attribute Undefined errors to.

        Subclasses (e.g. providers.ProviderContext) override to supply a model
        node; this base context has no node.
        """
        return None

    def _get_namespace_builder(self) -> MacroNamespaceBuilder:
        # avoid an import loop
        from dbt.adapters.factory import get_adapter_package_names

        internal_packages: List[str] = get_adapter_package_names(self.config.credentials.type)
        return MacroNamespaceBuilder(
            self.config.project_name,
            self.search_package,
            self.macro_stack,
            internal_packages,
            self._namespace_node(),
        )

    # This does not use the Mashumaro code
    def to_dict(self):
        dct = super().to_dict()
        # This moves all of the macros in the 'namespace' into top level
        # keys in the manifest dictionary
        if isinstance(self.namespace, TestMacroNamespace):
            dct.update(self.namespace.local_namespace)
            dct.update(self.namespace.project_namespace)
        elif isinstance(self.namespace, LazyMacroNamespace):
            proxy = MacroDictProxy(dct, self.namespace)
            # MacroGenerator wrappers must look up sibling macros (e.g.
            # ``{{ pkg.other_macro() }}`` from inside a macro body) via
            # the proxy so the lazy namespace gets consulted. The eager
            # MacroNamespace path achieves this by mutating ``_ctx`` in
            # ``dct.update(self.namespace)`` below; the lazy path swaps
            # the wrapper ctx instead so we don't materialize all macros.
            self.namespace.set_ctx(proxy)
            # Override-of-builtin macros must be written into the dict so
            # ``dict.__getitem__`` finds them before falling back to the
            # builtin. Must run AFTER set_ctx so the materialized wrappers
            # close over the proxy, not the bare ``_ctx``.
            proxy.materialize_builtin_overrides()
            return proxy
        else:
            dct.update(self.namespace)

        return dct

    @contextproperty()
    def context_macro_stack(self):
        return self.macro_stack
