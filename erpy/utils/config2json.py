import dataclasses
import json
from typing import Dict, Any


class Config2JSONEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)
        except TypeError:
            return str(o)


def config2json(config: Any) -> str:
    return json.dumps(obj=config, cls=Config2JSONEncoder)


def config2dict(config: Any) -> Dict:
    return json.loads(config2json(config=config))
