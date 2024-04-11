import os
from pathlib import Path
from typing import Optional, Union

import pytest
import typeguard
from parsl.data_provider.files import File

import psiflow
from psiflow.data import Dataset, NullState


def test_serial_simple(tmp_path):
    @psiflow.serializable
    class SomeSerial:
        pass

    @typeguard.typechecked
    class Test:
        foo: int
        bar: psiflow._DataFuture
        baz: Union[float, str]
        bam: Optional[SomeSerial]
        bao: SomeSerial
        bap: list[SomeSerial, ...]

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    new_cls = psiflow.serializable(Test)
    instance = new_cls(
        foo=3,
        bar=File("asdfl"),
        baz="asdflk",
        bam=None,
        bao=SomeSerial(),
        bap=[SomeSerial(), SomeSerial()],
    )
    assert instance.foo == 3
    assert instance._attrs["foo"] == 3

    # test independence
    instance._attrs["test"] = 1
    instance_ = new_cls(foo=4, bar=File("asdfl"))
    assert "test" not in instance_._attrs
    assert instance_.foo == 4
    assert instance.foo == 3

    assert tuple(instance._files.keys()) == ("bar",)
    assert tuple(instance._attrs.keys()) == ("foo", "baz", "test")
    assert tuple(instance._serial.keys()) == ("bam", "bao", "bap")
    assert type(instance._serial["bap"]) is list
    assert len(instance._serial["bap"]) == 2

    # serialization/deserialization of 'complex' Test instance
    json_dump = psiflow.serialize(instance).result()
    instance_ = psiflow.deserialize(json_dump, custom_cls=[new_cls, SomeSerial])

    assert instance.foo == instance_.foo
    assert instance.bar.filepath == instance_.bar.filepath
    assert instance.baz == instance_.baz
    assert instance.bam == instance_.bam
    assert type(instance_.bao) is SomeSerial
    assert len(instance_.bap) == 2
    assert type(instance_.bap[0]) is SomeSerial
    assert type(instance_.bap[1]) is SomeSerial
    assert id(instance) != id(instance_)

    # check classes created before test execution, e.g. Dataset
    data = Dataset([NullState])
    assert "data_future" in data._files
    assert len(data._attrs) == 0
    assert len(data._serial) == 0
    with pytest.raises(typeguard.TypeCheckError):  # try something stupid
        data.data_future = 0

    # test getter / setter
    data.data_future = File("some_file")
    assert type(data.data_future) is File

    # test basic serialization
    dumped_json = psiflow.serialize(data).result()
    assert "Dataset" in dumped_json
    assert len(dumped_json["Dataset"]["_attrs"]) == 0
    assert len(dumped_json["Dataset"]["_serial"]) == 0
    assert len(dumped_json["Dataset"]["_files"]) == 1
    assert dumped_json["Dataset"]["_files"]["data_future"] == data.data_future.filepath

    # test copy_to serialization
    data = Dataset([NullState])
    data.data_future.result()
    filename = Path(data.data_future.filepath).name
    assert os.path.exists(data.data_future.filepath)
    dumped_json = psiflow.serialize(data, copy_to=tmp_path / "test").result()
    os.remove(data.data_future.filepath)
    assert (tmp_path / "test").exists()
    assert (tmp_path / "test" / filename).exists()  # new file
