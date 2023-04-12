import importlib
from typing import Optional, Type

from loguru import logger
from .sym import sym_tbl


class Registrable:
    @classmethod
    def register(cls, alias: Optional[str] = None, exist_ok: bool = False):
        if cls not in sym_tbl().registry:
            sym_tbl().registry[cls] = {}
        registry = sym_tbl().registry[cls]
        if alias is not None and '.' in alias:
            # 需要更强的规则吗？
            raise NameError("Invalid alias {}".format(alias))

        def add_subclass_to_registry(subclass: Type) -> Type:
            names2add = [subclass.__module__ + "." + subclass.__name__]
            if alias is not None and alias != subclass.__name__:
                names2add.append(subclass.__module__ + "." + alias)
            # Add to registry, raise an error if key has already been used.
            for name2add in names2add:
                if name2add in registry:
                    if exist_ok:
                        message = (
                            f"{name2add} has already been registered as {registry[name2add].__name__}, but "
                            f"exist_ok=True, so overwriting with {cls.__name__}"
                        )
                        logger.info(message)
                    else:
                        message = (
                            f"Cannot register {name2add} as {cls.__name__}; "
                            f"name already in use for {registry[name2add].__name__}"
                        )
                        raise KeyError(message)
                registry[name2add] = subclass
            return subclass

        return add_subclass_to_registry

    @classmethod
    def resolve_registered_module(cls, name: str):
        fullname = name
        if "." in name:
            parts = fullname.split(".")
            submodule = ".".join(parts[:-1])
            name = parts[-1]
        else:
            submodule = ""

        if cls in sym_tbl().registry:
            if fullname in sym_tbl().registry[cls]:
                return sym_tbl().registry[cls][fullname]
            else:
                mangled_name1 = "__main__." + name
                mangled_name2 = "__mp_main__." + name
                registered_module_cls = None
                if mangled_name1 in sym_tbl().registry[cls]:
                    registered_module_cls = sym_tbl().registry[cls][mangled_name1]
                elif mangled_name2 in sym_tbl().registry[cls]:
                    registered_module_cls = sym_tbl().registry[cls][mangled_name2]
                if registered_module_cls is not None:
                    # 有一种特殊情况，如果注册模块的文件是入口文件，那么注册表中对应的可能是__main__.name或者__mp_main__.name
                    # 这个时需要尝试去进行匹配，但可能匹配的并不对，需要warning
                    logger.warning(
                        f"Module path not match for {fullname}, "
                        f"the registered module {registered_module_cls} might not be what you want"
                    )
                    return registered_module_cls

        # try import if not hit
        # 在import的时候register执行了
        _ = importlib.import_module(submodule)

        if cls not in sym_tbl().registry:
            raise RuntimeError(
                f"{cls.__name__} has not been registered yet, "
                f"available types = {[c.__name__ for c in sym_tbl().registry]}"
            )
        elif fullname not in sym_tbl().registry[cls]:
            available = list(sym_tbl().registry[cls].keys())
            raise KeyError(
                f"'{fullname}' is not a registered name for '{cls.__name__}, availables modules = {available}'"
            )
        else:
            return sym_tbl().registry[cls][fullname]
