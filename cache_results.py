from pathlib import Path
import pandas as pd
from collections.abc import Callable
from hashlib import md5
from inspect import currentframe, signature


def cache_pandas(fun=None, path: Path = Path("."), pd_varnames: bool = False):
    def cached_values(fun):
        def newfun(*args, **kwargs):
            global callers_local_vars
            sig = signature(fun)

            if sig.return_annotation is not pd.DataFrame:
                raise TypeError(
                    "The wrapped function must have a return type of pandas DataFrame "
                    "and be annotated as so."
                )

            bound_args = sig.bind(*args, **kwargs)
            all_args = {
                **dict(
                    (param.name, param.default)
                    for param in sig.parameters.values()
                ),
                **dict(bound_args.arguments.items()),
            }

            callers_local_vars = currentframe().f_back.f_locals.items()

            args_ = list()
            for value in all_args.values():
                if isinstance(value, pd.DataFrame) or (
                    hasattr(value, "data")
                    and isinstance(value.data, pd.DataFrame)
                ):
                    attempt = None
                    if pd_varnames:
                        attempt = next(
                            (
                                var_name
                                for var_name, var_val in callers_local_vars
                                if var_val is value
                            ),
                            None,
                        )

                    if attempt is not None:
                        args_.append(attempt)
                    else:
                        args_.append(md5(value.values.tobytes()).hexdigest())
                else:
                    args_.append(f"{value}")

            filepath = path / (fun.__name__ + "_" + "_".join(args_) + ".pkl")

            if filepath.exists():
                print(f"Reading cached data from {filepath}")
                return pd.read_pickle(filepath)

            res = fun(*args, **kwargs)
            res.to_pickle(filepath)
            return res

        return newfun

    if isinstance(fun, Callable):
        return cached_values(fun)
    else:
        return cached_values
