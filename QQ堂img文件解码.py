# -*- coding: utf-8 -*-
from __future__ import annotations
import ctypes, os
from PIL import Image


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

    @property
    def n_frames_per_direction(self):
        tmp = self.n_total_frames // self.n_directions
        assert self.n_total_frames == tmp * self.n_directions
        return tmp


SPRITE_METADATA_SIZE = ctypes.sizeof(QQSpriteMetadata)


class QQSpriteFrameMetadata(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),  # possible values: 0
        ("frame_info_cx", ctypes.c_int32),
        ("frame_info_cy", ctypes.c_int32),
        ("mode", ctypes.c_uint32),  # possible values: 285212672, 0, 3, 8, 16, in majority of cases it is 3 but sometimes it returns 8
    ]


SPRITE_FRAME_METADATA_SIZE = ctypes.sizeof(QQSpriteFrameMetadata)


class QQImageMetadata(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("unknown", ctypes.c_uint32),
    ]


IMAGE_METADATA_SIZE = ctypes.sizeof(QQImageMetadata)


class QQTImage(object):
    def __init__(self, header: QQImageMetadata, data, mode):
        self.width = header.width
        self.height = header.height
        self.data = data
        self.mode = mode

    def tostring(self):
        return self.data


class QQTSprite(object):
    def __init__(self, metadata: QQSpriteMetadata):
        assert metadata.magic_1 == b'QQF\x1aDIMG'
        assert metadata.version_id == 65536 or metadata.version_id == 65537
        assert metadata.frame_info_instance_size == 24
        self.metadata = metadata
        self.frames = [[] for i in range(metadata.n_directions)]

    def add_frame(self, direction, image: QQTImage):
        self.frames[direction].append(image)

    def save_to_dir(self, directory: str):
        for direction, frames_per_direction in enumerate(self.frames):
            for index, image in enumerate(frames_per_direction):
                # Image.frombuffer receives bytes with ordering top-left to bottom-right each pixel four bytes (<red 0-255>, <green 0-255>, <blue 0-255>, <alpha 0-255>)
                img = Image.frombuffer(image.mode, (image.width, image.height), image.data, "raw")
                img.save(os.path.join(directory, str(direction) + "_" + str(index) + ".png"))

    @classmethod
    def decode_rgb565_alpha_channel(cls, byte):
        return int(byte * 255.0 / 32.0)

    @classmethod
    def decode_rgb565_red_channel(cls, higher_byte, lower_byte):
        return int(((higher_byte & 0b11111000) >> 3) * 255.0 / 31.0)

    @classmethod
    def decode_rgb565_green_channel(cls, higher_byte, lower_byte):
        return int((((higher_byte & 0b111) << 3) | ((lower_byte & 0b11100000) >> 5)) * 255.0 / 63.0)

    @classmethod
    def decode_rgb565_blue_channel(cls, higher_byte, lower_byte):
        return int((lower_byte & 0b11111) * 255.0 / 31.0)

    @classmethod
    def decode_rgb565_with_alpha_channel_data(cls, width, height, original_image_data: bytes):
        standard_pixel_array = bytearray(width * height * 4)
        for i in range(height):
            for j in range(width):
                # little-endian
                lower_byte = original_image_data[(i * width + j) * 2]
                higher_byte = original_image_data[(i * width + j) * 2 + 1]
                standard_pixel_array_start_offset = (i * width + j) * 4
                standard_pixel_array[standard_pixel_array_start_offset] = cls.decode_rgb565_red_channel(higher_byte, lower_byte)
                standard_pixel_array[standard_pixel_array_start_offset + 1] = cls.decode_rgb565_green_channel(higher_byte, lower_byte)
                standard_pixel_array[standard_pixel_array_start_offset + 2] = cls.decode_rgb565_blue_channel(higher_byte, lower_byte)
                # alpha channel is stored alone at the end
                standard_pixel_array[standard_pixel_array_start_offset + 3] = cls.decode_rgb565_alpha_channel(
                    original_image_data[width * height * 2 + i * width + j])
        return bytes(standard_pixel_array)

    @classmethod
    def decode_rgb32_data(cls, width, height, original_image_data: bytes):
        standard_pixel_array = bytearray(width * height * 4)
        for i in range(height):
            for j in range(width):
                original_image_data_start_offset = (i * width + j) * 3
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

    @classmethod
    def open(cls, filepath: str) -> QQTSprite:
        assert filepath.endswith(".img")
        with open(filepath, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_length = f.tell()
            f.seek(0, os.SEEK_SET)
            sprite = QQTSprite(QQSpriteMetadata.from_buffer_copy(f.read(SPRITE_METADATA_SIZE)))
            for direction in range(sprite.metadata.n_directions):
                for frame in range(sprite.metadata.n_frames_per_direction):
                    frame_metadata = QQSpriteFrameMetadata.from_buffer_copy(f.read(SPRITE_FRAME_METADATA_SIZE))
                    assert frame_metadata.magic == 0
                    if frame_metadata.mode != 0:
                        image_metadata = QQImageMetadata.from_buffer_copy(f.read(IMAGE_METADATA_SIZE))
                        if frame_metadata.mode == 3 or frame_metadata.mode == 285212672:  # RGB565 + alpha channel at the end
                            data_size = image_metadata.width * image_metadata.height * 3  # 2 bytes for RGB565 and 1 byte for alpha channel
                            assert data_size < file_length
                            original_image_data = f.read(data_size)
                            sprite.add_frame(direction,
                                             QQTImage(image_metadata, cls.decode_rgb565_with_alpha_channel_data(image_metadata.width, image_metadata.height,
                                                                                                                original_image_data), "RGBA"))
                        elif frame_metadata.mode == 8:  # RGB32
                            data_size = image_metadata.width * image_metadata.height * 4
                            assert data_size < file_length
                            original_image_data = f.read(data_size)
                            sprite.add_frame(direction,
                                             QQTImage(image_metadata, cls.decode_rgb32_data(image_metadata.width, image_metadata.height, original_image_data),
                                                      "RGBA"))
                        elif frame_metadata.mode == 16:  # RGB24
                            data_size = image_metadata.width * image_metadata.height * 3
                            assert data_size < file_length
                            original_image_data = f.read(data_size)
                            sprite.add_frame(direction,
                                             QQTImage(image_metadata, cls.decode_rgb24_data(image_metadata.width, image_metadata.height,
                                                                                            original_image_data), "RGB"))
                        else:
                            raise AssertionError("error format when reading " + filepath + " because mode is " + str(frame_metadata.mode))
            total_read_length = f.tell()
            assert file_length == total_read_length, filepath
            return sprite


def convert_img_file_to_png(img_file_path, output_dir):
    sprite = QQTSprite.open(img_file_path)
    output_dir = os.path.join(output_dir, os.path.splitext(img_file_path)[0])
    os.makedirs(output_dir, exist_ok=True)
    sprite.save_to_dir(output_dir)


def find_img_file_in_folder_and_convert_files_to_png(input_dir, output_dir):
    for file in os.listdir(input_dir):
        file = os.path.join(input_dir, file)
        if os.path.isdir(file):
            find_img_file_in_folder_and_convert_files_to_png(file, output_dir)
        else:
            if file.endswith(".img"):
                try:
                    convert_img_file_to_png(file, output_dir)
                except Exception as e:
                    print("ignored " + file + " because of " + str(e))


if __name__ == '__main__':
    find_img_file_in_folder_and_convert_files_to_png("input", "output")