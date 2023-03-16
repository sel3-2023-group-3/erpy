import dataclasses
import json
from typing import Dict, TypeVar, Any, dataclass_transform, Type


class Config2JSONEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)
        except TypeError:
            return str(o)


T = TypeVar("T")


@dataclass_transform(
    field_specifiers=(dataclasses.Field, dataclasses.field)
)
def config2json(config: Type[T]) -> str:
    return json.dumps(obj=config, cls=Config2JSONEncoder)


@dataclass_transform(
    field_specifiers=(dataclasses.Field, dataclasses.field)
)
def config2dict(config: Type[T]) -> Dict:
    return json.loads(config2json(config=config))
