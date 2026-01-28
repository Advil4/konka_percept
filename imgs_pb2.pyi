from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Img(_message.Message):
    __slots__ = ("timeStamp", "img", "requiresMasks")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    IMG_FIELD_NUMBER: _ClassVar[int]
    REQUIRESMASKS_FIELD_NUMBER: _ClassVar[int]
    timeStamp: bytes
    img: bytes
    requiresMasks: bool
    def __init__(self, timeStamp: _Optional[bytes] = ..., img: _Optional[bytes] = ..., requiresMasks: bool = ...) -> None: ...

class Tracks(_message.Message):
    __slots__ = ("timeStamp", "tracks")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    TRACKS_FIELD_NUMBER: _ClassVar[int]
    timeStamp: bytes
    tracks: bytes
    def __init__(self, timeStamp: _Optional[bytes] = ..., tracks: _Optional[bytes] = ...) -> None: ...

class MasksTracks(_message.Message):
    __slots__ = ("timeStamp", "tracks")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    TRACKS_FIELD_NUMBER: _ClassVar[int]
    timeStamp: bytes
    tracks: _containers.RepeatedCompositeFieldContainer[TrackRow]
    def __init__(self, timeStamp: _Optional[bytes] = ..., tracks: _Optional[_Iterable[_Union[TrackRow, _Mapping]]] = ...) -> None: ...

class TrackRow(_message.Message):
    __slots__ = ("x1", "y1", "x2", "y2", "trackId", "conf", "cls", "mask", "isMove", "feats")
    X1_FIELD_NUMBER: _ClassVar[int]
    Y1_FIELD_NUMBER: _ClassVar[int]
    X2_FIELD_NUMBER: _ClassVar[int]
    Y2_FIELD_NUMBER: _ClassVar[int]
    TRACKID_FIELD_NUMBER: _ClassVar[int]
    CONF_FIELD_NUMBER: _ClassVar[int]
    CLS_FIELD_NUMBER: _ClassVar[int]
    MASK_FIELD_NUMBER: _ClassVar[int]
    ISMOVE_FIELD_NUMBER: _ClassVar[int]
    FEATS_FIELD_NUMBER: _ClassVar[int]
    x1: float
    y1: float
    x2: float
    y2: float
    trackId: str
    conf: float
    cls: str
    mask: _containers.RepeatedCompositeFieldContainer[Masks]
    isMove: bool
    feats: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, x1: _Optional[float] = ..., y1: _Optional[float] = ..., x2: _Optional[float] = ..., y2: _Optional[float] = ..., trackId: _Optional[str] = ..., conf: _Optional[float] = ..., cls: _Optional[str] = ..., mask: _Optional[_Iterable[_Union[Masks, _Mapping]]] = ..., isMove: bool = ..., feats: _Optional[_Iterable[float]] = ...) -> None: ...

class Masks(_message.Message):
    __slots__ = ("x", "y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: int
    y: int
    def __init__(self, x: _Optional[int] = ..., y: _Optional[int] = ...) -> None: ...
