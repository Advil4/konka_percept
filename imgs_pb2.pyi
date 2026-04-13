from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ClientRequest(_message.Message):
    __slots__ = ("percept_enable", "percept_enable_visual", "percept_enable_img_save", "percept_enable_visual_save", "percept_enable_result_save", "percept_enable_img_save_path", "percept_enable_visual_save_path", "percept_enable_result_save_path")
    PERCEPT_ENABLE_FIELD_NUMBER: _ClassVar[int]
    PERCEPT_ENABLE_VISUAL_FIELD_NUMBER: _ClassVar[int]
    PERCEPT_ENABLE_IMG_SAVE_FIELD_NUMBER: _ClassVar[int]
    PERCEPT_ENABLE_VISUAL_SAVE_FIELD_NUMBER: _ClassVar[int]
    PERCEPT_ENABLE_RESULT_SAVE_FIELD_NUMBER: _ClassVar[int]
    PERCEPT_ENABLE_IMG_SAVE_PATH_FIELD_NUMBER: _ClassVar[int]
    PERCEPT_ENABLE_VISUAL_SAVE_PATH_FIELD_NUMBER: _ClassVar[int]
    PERCEPT_ENABLE_RESULT_SAVE_PATH_FIELD_NUMBER: _ClassVar[int]
    percept_enable: bool
    percept_enable_visual: bool
    percept_enable_img_save: bool
    percept_enable_visual_save: bool
    percept_enable_result_save: bool
    percept_enable_img_save_path: str
    percept_enable_visual_save_path: str
    percept_enable_result_save_path: str
    def __init__(self, percept_enable: bool = ..., percept_enable_visual: bool = ..., percept_enable_img_save: bool = ..., percept_enable_visual_save: bool = ..., percept_enable_result_save: bool = ..., percept_enable_img_save_path: _Optional[str] = ..., percept_enable_visual_save_path: _Optional[str] = ..., percept_enable_result_save_path: _Optional[str] = ...) -> None: ...

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

class CommandStatus(_message.Message):
    __slots__ = ("command", "message")
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    command: str
    message: str
    def __init__(self, command: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class MasksTracks(_message.Message):
    __slots__ = ("timeStamp", "tracks", "command_statuses")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    TRACKS_FIELD_NUMBER: _ClassVar[int]
    COMMAND_STATUSES_FIELD_NUMBER: _ClassVar[int]
    timeStamp: bytes
    tracks: _containers.RepeatedCompositeFieldContainer[TrackRow]
    command_statuses: _containers.RepeatedCompositeFieldContainer[CommandStatus]
    def __init__(self, timeStamp: _Optional[bytes] = ..., tracks: _Optional[_Iterable[_Union[TrackRow, _Mapping]]] = ..., command_statuses: _Optional[_Iterable[_Union[CommandStatus, _Mapping]]] = ...) -> None: ...
