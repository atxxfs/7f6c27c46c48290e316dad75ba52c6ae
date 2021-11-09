# -*- coding: utf-8 -*-
from __future__ import annotations

import abc
import ctypes
import functools
import os
import typing

from PIL import Image


class QQTImageMetadata(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("unknown", ctypes.c_uint32),
    ]


QQT_IMAGE_METADATA_SIZE = ctypes.sizeof(QQTImageMetadata)


class QQTImage(abc.ABC):
    @property
    @abc.abstractmethod
    def width(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def height(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def instance_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def mode(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def standard_image_data(self) -> bytes:
        pass

    @abc.abstractmethod
    def save(self, filepath):
        pass


class QQTEmptyImage(QQTImage):
    @property
    def width(self) -> int:
        return 0

    @property
    def height(self) -> int:
        return 0

    @property
    def instance_size(self) -> int:
        return 0

    @property
    def mode(self) -> str:
        return "RGB"

    @property
    def standard_image_data(self) -> bytes:
        return b""

    def save(self, filepath):
        pass


class QQTNonEmptyImage(QQTImage, metaclass=abc.ABCMeta):
    def __init__(self, metadata: QQTImageMetadata, original_image_data):
        self._metadata = metadata
        self._original_image_data = original_image_data

    @property
    def width(self) -> int:
        return self._metadata.width

    @property
    def height(self) -> int:
        return self._metadata.height

    @functools.cached_property
    def instance_size(self) -> int:
        return QQT_IMAGE_METADATA_SIZE + len(self._original_image_data)

    def save(self, filepath):
        # Image.frombuffer receives bytes with ordering top-left to bottom-right each pixel four bytes (<red 0-255>, <green 0-255>, <blue 0-255>, <alpha 0-255>)
        img = Image.frombuffer(self.mode, (self.width, self.height), self.standard_image_data, "raw")
        img.save(filepath)


class QQTRGBAImage(QQTNonEmptyImage, metaclass=abc.ABCMeta):
    def __init__(self, metadata: QQTImageMetadata, original_image_data):
        super().__init__(metadata, original_image_data)

    @property
    def mode(self) -> str:
        return "RGBA"


class QQTRGB565WithAlphaChannelAtTheEndImage(QQTRGBAImage):
    def __init__(self, metadata: QQTImageMetadata, original_image_data):
        super().__init__(metadata, original_image_data)

    @classmethod
    def decode_rgb565_alpha_channel(cls, byte):
        return int(byte * 255.0 / 32.0)

    @classmethod
    def decode_rgb565_red_channel(cls, higher_byte):
        return int(((higher_byte & 0b11111000) >> 3) * 255.0 / 31.0)

    @classmethod
    def decode_rgb565_green_channel(cls, higher_byte, lower_byte):
        return int((((higher_byte & 0b111) << 3) | ((lower_byte & 0b11100000) >> 5)) * 255.0 / 63.0)

    @classmethod
    def decode_rgb565_blue_channel(cls, lower_byte):
        return int((lower_byte & 0b11111) * 255.0 / 31.0)

    @classmethod
    def decode_rgb565_with_alpha_channel_at_the_end_data(cls, width, height, original_image_data: bytes) -> bytes:
        standard_pixel_array = bytearray(width * height * 4)
        for i in range(height):
            for j in range(width):
                # little-endian
                original_image_data_start_offset = (i * width + j) * 2
                lower_byte = original_image_data[original_image_data_start_offset]
                higher_byte = original_image_data[original_image_data_start_offset + 1]
                standard_pixel_array_start_offset = (i * width + j) * 4
                standard_pixel_array[standard_pixel_array_start_offset] = cls.decode_rgb565_red_channel(higher_byte)
                standard_pixel_array[standard_pixel_array_start_offset + 1] = cls.decode_rgb565_green_channel(higher_byte, lower_byte)
                standard_pixel_array[standard_pixel_array_start_offset + 2] = cls.decode_rgb565_blue_channel(lower_byte)
                # alpha channel is stored alone at the end
                original_image_data_alpha_channel_offset = width * height * 2 + i * width + j
                standard_pixel_array[standard_pixel_array_start_offset + 3] = cls.decode_rgb565_alpha_channel(
                    original_image_data[original_image_data_alpha_channel_offset])
        return bytes(standard_pixel_array)

    @functools.cached_property
    def standard_image_data(self) -> bytes:
        return self.__class__.decode_rgb565_with_alpha_channel_at_the_end_data(self.width, self.height, self._original_image_data)

    @classmethod
    def from_buffer(cls, source: bytes, offset: int) -> QQTRGB565WithAlphaChannelAtTheEndImage:
        metadata = QQTImageMetadata.from_buffer_copy(source, offset)
        offset += QQT_IMAGE_METADATA_SIZE
        data_size = metadata.width * metadata.height * 3  # 2 bytes for RGB565 and 1 byte for alpha channel
        original_image_data = source[offset:offset + data_size]
        assert len(original_image_data) == data_size
        return QQTRGB565WithAlphaChannelAtTheEndImage(metadata, original_image_data)


class QQTRGB32Image(QQTRGBAImage):
    def __init__(self, metadata: QQTImageMetadata, original_image_data):
        super().__init__(metadata, original_image_data)

    @classmethod
    def decode_rgb32_data(cls, width, height, original_image_data: bytes):
        standard_pixel_array = bytearray(width * height * 4)
        for i in range(height):
            for j in range(width):
                original_image_data_start_offset = (i * width + j) * 4
                red = original_image_data[original_image_data_start_offset + 2]
                green = original_image_data[original_image_data_start_offset + 1]
                blue = original_image_data[original_image_data_start_offset]
                alpha = original_image_data[original_image_data_start_offset + 3]
                standard_pixel_array_start_offset = (i * width + j) * 4
                standard_pixel_array[standard_pixel_array_start_offset] = red
                standard_pixel_array[standard_pixel_array_start_offset + 1] = green
                standard_pixel_array[standard_pixel_array_start_offset + 2] = blue
                standard_pixel_array[standard_pixel_array_start_offset + 3] = alpha
        return bytes(standard_pixel_array)

    @functools.cached_property
    def standard_image_data(self) -> bytes:
        return self.__class__.decode_rgb32_data(self.width, self.height, self._original_image_data)

    @classmethod
    def from_buffer(cls, source: bytes, offset: int) -> QQTRGB32Image:
        metadata = QQTImageMetadata.from_buffer_copy(source, offset)
        offset += QQT_IMAGE_METADATA_SIZE
        data_size = metadata.width * metadata.height * 4
        original_image_data = source[offset:offset + data_size]
        assert len(original_image_data) == data_size
        return QQTRGB32Image(metadata, original_image_data)


class QQTRGB24Image(QQTNonEmptyImage):
    def __init__(self, metadata: QQTImageMetadata, original_image_data):
        super().__init__(metadata, original_image_data)

    @property
    def mode(self) -> str:
        return "RGB"

    @classmethod
    def decode_rgb24_data(cls, width, height, original_image_data: bytes):
        standard_pixel_array = bytearray(width * height * 3)
        for i in range(height):
            for j in range(width):
                original_image_data_start_offset = (i * width + j) * 3
                red = original_image_data[original_image_data_start_offset + 2]
                green = original_image_data[original_image_data_start_offset + 1]
                blue = original_image_data[original_image_data_start_offset]
                standard_pixel_array_start_offset = (i * width + j) * 3
                standard_pixel_array[standard_pixel_array_start_offset] = red
                standard_pixel_array[standard_pixel_array_start_offset + 1] = green
                standard_pixel_array[standard_pixel_array_start_offset + 2] = blue
        return bytes(standard_pixel_array)

    @functools.cached_property
    def standard_image_data(self) -> bytes:
        return self.__class__.decode_rgb24_data(self.width, self.height, self._original_image_data)

    @classmethod
    def from_buffer(cls, source: bytes, offset: int) -> QQTRGB24Image:
        metadata = QQTImageMetadata.from_buffer_copy(source, offset)
        offset += QQT_IMAGE_METADATA_SIZE
        data_size = metadata.width * metadata.height * 3
        original_image_data = source[offset:offset + data_size]
        assert len(original_image_data) == data_size
        return QQTRGB24Image(metadata, original_image_data)


class QQTSpriteFrameMetadata(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),  # possible values: 0
        ("frame_info_cx", ctypes.c_int32),
        ("frame_info_cy", ctypes.c_int32),
        ("mode", ctypes.c_uint32),  # possible values: 285212672, 0, 3, 8, 16, in majority of cases it is 3 but sometimes it returns 8
    ]


QQT_SPRITE_FRAME_METADATA_SIZE = ctypes.sizeof(QQTSpriteFrameMetadata)


class QQTSpriteFrame(object):
    def __init__(self, metadata: QQTSpriteFrameMetadata, image: QQTImage):
        self._metadata = metadata
        self._image = image

    @functools.cached_property
    def instance_size(self) -> int:
        return QQT_SPRITE_FRAME_METADATA_SIZE + self._image.instance_size

    def save(self, filepath):
        self._image.save(filepath)

    @classmethod
    def from_buffer(cls, source: bytes, offset: int) -> QQTSpriteFrame:
        frame_metadata = QQTSpriteFrameMetadata.from_buffer_copy(source, offset)
        offset += QQT_SPRITE_FRAME_METADATA_SIZE
        assert frame_metadata.magic == 0
        if frame_metadata.mode == 0:
            return QQTSpriteFrame(frame_metadata, QQTEmptyImage())
        elif frame_metadata.mode == 3 or frame_metadata.mode == 285212672:
            return QQTSpriteFrame(frame_metadata, QQTRGB565WithAlphaChannelAtTheEndImage.from_buffer(source, offset))
        elif frame_metadata.mode == 8:
            return QQTSpriteFrame(frame_metadata, QQTRGB32Image.from_buffer(source, offset))
        elif frame_metadata.mode == 16:  # RGB24
            return QQTSpriteFrame(frame_metadata, QQTRGB24Image.from_buffer(source, offset))
        else:
            raise AssertionError("error format because mode is " + str(frame_metadata.mode))


class QQSpriteMetadata(ctypes.Structure):
    _fields_ = [
        ("magic_1", ctypes.c_char * 8),  # possible value: QQF.DIMG
        ("version_id", ctypes.c_uint32),  # possible values: 65536 or 65537
        ("frame_info_instance_size", ctypes.c_uint32),  # possible values: 24
        ("n_total_frames", ctypes.c_uint32),
        ("n_directions", ctypes.c_uint32),  # possible values: 1, 4
        ("unknown_1", ctypes.c_uint32),
        ("unknown_2", ctypes.c_uint32),
        ("width", ctypes.c_uint32),  # I guess this is width of sprite, but sometimes it doesn't match image width
        ("height", ctypes.c_uint32),  # I guess this is height of sprite, but sometimes it doesn't match image height
    ]

    @functools.cached_property
    def n_frames_per_direction(self) -> int:
        tmp = self.n_total_frames // self.n_directions
        assert self.n_total_frames == tmp * self.n_directions
        return tmp


QQT_SPRITE_METADATA_SIZE = ctypes.sizeof(QQSpriteMetadata)


class QQTSprite(object):
    def __init__(self, metadata: QQSpriteMetadata, frames: typing.List[typing.List[QQTSpriteFrame]]):
        self._metadata = metadata
        self._frames = frames

    @functools.cached_property
    def instance_size(self):
        size = QQT_SPRITE_METADATA_SIZE
        for frames_per_direction in self._frames:
            for frame in frames_per_direction:
                size += frame.instance_size
        return size

    def save(self, prefix: str):
        for direction, frames_per_direction in enumerate(self._frames):
            for index, frame in enumerate(frames_per_direction):
                frame.save(prefix + "_" + str(direction) + "_" + str(index) + ".png")

    @classmethod
    def from_buffer(cls, source: bytes, offset: int = 0) -> QQTSprite:
        metadata = QQSpriteMetadata.from_buffer_copy(source, offset)
        assert metadata.magic_1 == b'QQF\x1aDIMG'
        assert metadata.version_id == 65536 or metadata.version_id == 65537
        assert metadata.frame_info_instance_size == 24
        offset += QQT_SPRITE_METADATA_SIZE
        frames: typing.List[typing.List[QQTSpriteFrame]] = []
        for direction in range(metadata.n_directions):
            frames_per_direction: typing.List[QQTSpriteFrame] = []
            for frame in range(metadata.n_frames_per_direction):
                sprite_frame = QQTSpriteFrame.from_buffer(source, offset)
                offset += sprite_frame.instance_size
                frames_per_direction.append(sprite_frame)
            frames.append(frames_per_direction)
        return QQTSprite(metadata, frames)

    @classmethod
    def open(cls, filepath: str) -> QQTSprite:
        assert filepath.endswith(".img")
        with open(filepath, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_length = f.tell()
            f.seek(0, os.SEEK_SET)
            sprite = cls.from_buffer(f.read())
            assert file_length == sprite.instance_size
            return sprite


def convert_img_file_to_png(img_file_path, output_path_prefix):
    sprite = QQTSprite.open(img_file_path)
    output_path_prefix = os.path.join(output_path_prefix, os.path.splitext(img_file_path)[0])
    output_dir = os.path.dirname(output_path_prefix)
    os.makedirs(output_dir, exist_ok=True)
    sprite.save(output_path_prefix)


def find_img_files_in_folder_and_convert_files_to_png(input_dir, output_dir):
    for file in os.listdir(input_dir):
        file = os.path.join(input_dir, file)
        if os.path.isdir(file):
            find_img_files_in_folder_and_convert_files_to_png(file, output_dir)
        else:
            if file.endswith(".img"):
                try:
                    convert_img_file_to_png(file, output_dir)
                except Exception as e:
                    print("ignored " + file + " because of " + str(e))


if __name__ == '__main__':
    find_img_files_in_folder_and_convert_files_to_png("input", "output")