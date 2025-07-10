"""
    Generate Image CAPTCHAs, just the normal image CAPTCHAs you are using.
    Modification of: https://github.com/lepture/captcha/blob/main/src/captcha/image.py
"""

from __future__ import annotations

# Generic 
import os
import random
import secrets
import typing as t
from io import BytesIO
from typing import Tuple, List

# PIL
from PIL import ImageFont
from PIL.Image import new as createImage, Image, Transform, Resampling, blend
from PIL.ImageFont import FreeTypeFont, truetype
from PIL.ImageDraw import Draw, ImageDraw
from PIL.ImageFilter import SMOOTH

# Type aliases
__all__ = ['ImageCaptcha']

# Special
ColorTuple = t.Union[t.Tuple[int, int, int], t.Tuple[int, int, int, int]]
DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
DEFAULT_FONTS = [os.path.join(DATA_DIR, 'DroidSansMono.ttf')]



class ImageCaptcha:
    """Create an image CAPTCHA.

    Many of the codes are borrowed from wheezy.captcha, with a modification
    for memory and developer friendly.

    ImageCaptcha has one built-in font, DroidSansMono, which is licensed under
    Apache License 2. You should always use your own fonts::

        captcha = ImageCaptcha(fonts=['/path/to/A.ttf', '/path/to/B.ttf'])

    You can put as many fonts as you like. But be aware of your memory, all of
    the fonts are loaded into your memory, so keep them a lot, but not too
    many.

    :param width: The width of the CAPTCHA image.
    :param height: The height of the CAPTCHA image.
    :param fonts: Fonts to be used to generate CAPTCHA images.
    :param font_sizes: Random choose a font size from this parameters.
    :param gaussian_noise_rate: The rate of blending for gaussian noise.
    """
    lookup_table: list[int] = [int(i * 1.97) for i in range(256)]

    def __init__(
            self,
            # resolution
            width: int = 256,
            height: int = 128,
            # distortion
            character_offset_dx: tuple[int, int] = (0, 4),
            character_offset_dy: tuple[int, int] = (0, 6),
            character_rotate: tuple[int, int] = (-30, 30),
            character_warp_dx: tuple[float, float] = (0.1, 0.3),
            character_warp_dy: tuple[float, float] = (0.2, 0.3),
            word_space_probability: float = 0.5,
            word_offset_dx: float = 0.25,
            # fonts
            fonts: list[str] | None = None,
            font_sizes: tuple[int, ...] | None = None,
            # difficulty
            difficulty: str = "hard",
            max_gaussian_noise_rate: float = 0.35
    ):
        # resolution
        self._width = width
        self._height = height
        # distortion
        self.character_offset_dx = character_offset_dx
        self.character_offset_dy = character_offset_dy
        self.character_rotate = character_rotate
        self.character_warp_dx = character_warp_dx
        self.character_warp_dy = character_warp_dy
        self.word_space_probability = word_space_probability
        self.word_offset_dx = word_offset_dx
        # fonts
        self._fonts = fonts or DEFAULT_FONTS
        self._font_sizes = font_sizes or (42, 50, 56)
        self._truefonts: list[FreeTypeFont] = []
        # difficulty
        self._difficulty = difficulty
        self._max_gaussian_noise_rate = max_gaussian_noise_rate


    @property
    def truefonts(self) -> list[FreeTypeFont]:
        if self._truefonts:
            return self._truefonts
        self._truefonts = [
            truetype(n, s)
            for n in self._fonts
            for s in self._font_sizes
        ]
        return self._truefonts


    @staticmethod
    def create_noise_curve(image: Image, color: ColorTuple) -> Image:
        w, h = image.size
        # OLD
        # x1 = secrets.randbelow(int(w / 5) + 1) 
        # x2 = secrets.randbelow(w - int(w / 5) + 1) + int(w / 5)
        # y1 = secrets.randbelow(h - 2 * int(h / 5) + 1) + int(h / 5)
        # y2 = secrets.randbelow(h - y1 - int(h / 5) + 1) + y1
        # NEW 
        # Width-wise: x1 in [10%, 20%], x2 in [70%, 90%]
        x1 = secrets.randbelow(int(w * 0.1)) + int(w * 0.1)     # 10–20%
        x2 = secrets.randbelow(int(w * 0.2)) + int(w * 0.7)     # 70–90%
        # Height-wise: y1 and y2 in middle 40–60%
        y1 = secrets.randbelow(int(h * 0.1)) + int(h * 0.4)     # 30-40% (fr top)
        y2 = secrets.randbelow(int(h * 0.1)) + int(h * 0.501)   # 50-50% (fr top)
        points = [x1, y1, x2, y2]
        # Slight arc: small angle difference (e.g., 10–30 degrees)
        start = secrets.randbelow(21)
        end = secrets.randbelow(41) + 100
        # Draw
        Draw(image).arc(points, start, end, fill=color)
        return image


    @staticmethod
    def create_noise_dots(
            image: Image,
            color: ColorTuple,
            width: int = 3,
            number: int = 30) -> Image:
        draw = Draw(image)
        w, h = image.size
        while number:
            x1 = secrets.randbelow(w + 1)
            y1 = secrets.randbelow(h + 1)
            draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=width)
            number -= 1
        return image


    @staticmethod
    def create_gaussian_noise(image: Image, rate: float) -> Image:
        """Add gaussian noise to the image."""
        w, h = image.size
        # Create a new image with random pixel values
        noise = createImage('RGB', (w, h))
        pixels = noise.load()
        for i in range(w):
            for j in range(h):
                pixels[i, j] = (
                    secrets.randbelow(256),
                    secrets.randbelow(256),
                    secrets.randbelow(256)
                )
        # Blend the noise with the original image
        return blend(image, noise, rate)
    

    def _draw_character(
            self,
            c: str,
            draw: ImageDraw,
            color: ColorTuple) -> Image:
        font = secrets.choice(self.truefonts)
        # The multiline_textbbox is used to get the bounding box of the text
        # It's more accurate than textbbox for some fonts.
        # We use a dummy coordinate (1,1) as it doesn't affect the size.
        _, _, w, h = draw.multiline_textbbox((1, 1), c, font=font)

        dx1 = secrets.randbelow(self.character_offset_dx[1] - self.character_offset_dx[0] + 1) + self.character_offset_dx[0]
        dy1 = secrets.randbelow(self.character_offset_dy[1] - self.character_offset_dy[0] + 1) + self.character_offset_dy[0]
        
        # Create an image for the single character
        im = createImage('RGBA', (int(w) + dx1, int(h) + dy1))
        Draw(im).text((dx1, dy1), c, font=font, fill=color)

        # rotate
        im = im.crop(im.getbbox())
        im = im.rotate(
            self.character_rotate[0] + (secrets.randbits(32) / (2**32)) * (self.character_rotate[1] - self.character_rotate[0]),
            Resampling.BILINEAR,
            expand=True,
        )

        # warp
        dx2 = w * (secrets.randbits(32) / (2**32)) * (self.character_warp_dx[1] - self.character_warp_dx[0]) + self.character_warp_dx[0]
        dy2 = h * (secrets.randbits(32) / (2**32)) * (self.character_warp_dy[1] - self.character_warp_dy[0]) + self.character_warp_dy[0]
        x1 = int(secrets.randbits(32) / (2**32) * (dx2 - (-dx2)) + (-dx2))
        y1 = int(secrets.randbits(32) / (2**32) * (dy2 - (-dy2)) + (-dy2))
        x2 = int(secrets.randbits(32) / (2**32) * (dx2 - (-dx2)) + (-dx2))
        y2 = int(secrets.randbits(32) / (2**32) * (dy2 - (-dy2)) + (-dy2))
        w2 = w + abs(x1) + abs(x2)
        h2 = h + abs(y1) + abs(y2)
        data = (
            x1, y1,
            -x1, h2 - y2,
            w2 + x2, h2 + y2,
            w2 - x2, -y1,
        )
        im = im.resize((int(w2), int(h2)))
        im = im.transform((int(w), int(h)), Transform.QUAD, data)
        return im


    def _compose_image(
            self,
            base_img: Image.Image,
            char_images: List[Image.Image],
            text_width: int,
            num_chars: int,
            offset = True,
        ) -> Image.Image:
            """
            Paste a sequence of character images onto base_img,
            center them, apply random dx-offset, and resize if needed.
            """
            # ensure canvas wide enough
            canvas_width = max(text_width, self._width)
            img = base_img.resize((canvas_width, self._height))

            avg_char_w = text_width // num_chars
            max_dx = int(self.word_offset_dx * avg_char_w)
            side_space = max(self._width - text_width, 0)
            offset_x = side_space // 2

            for ch_img in char_images:
                w, h = ch_img.size
                mask = ch_img.convert('L').point(self.lookup_table)
                y = (self._height - h) // 2
                img.paste(ch_img, (offset_x, y), mask)

                # random shift within ±max_dx/2
                shift = secrets.randbelow(max_dx + 1) - (max_dx // 2)
                offset_x += w + shift

            # if we expanded beyond desired width, shrink back
            if canvas_width > self._width:
                img = img.resize((self._width, self._height))

            return img
    
    
    def create_captcha_image(
        self,
        chars: str,
        fg_color: ColorTuple,
        background: ColorTuple
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Create two CAPTCHA images:
         - image1: with gaussian noise and variable spacing
         - image2: clean (no noise) and fixed single-space between chars
        Returns (image1, image2).
        """
        # Base canvases
        base1 = createImage('RGB', (self._width, self._height), background)
        base2 = createImage('RGB', (self._width, self._height), background)

        # apply gaussian noise to base1 only
        if self._max_gaussian_noise_rate > 0:
            base1 = self.create_gaussian_noise(
                base1,
                round(random.uniform(0.2, self._max_gaussian_noise_rate), 3)
            )

        draw1 = Draw(base1)
        # draw2 = Draw(base2)

        # build per-character images
        imgs1: List[Image.Image] = []
        imgs2: List[Image.Image] = []

        for c in chars:
            char_color = random_color(10, 200, secrets.randbelow(36) + 220)

            # draw the character and a space
            img_char = self._draw_character(c, draw1, char_color)
            img_space = self._draw_character(" ", draw1, char_color)

            # for image1: sometimes skip or insert additional space
            if secrets.randbits(32) / (2**32) < self.word_space_probability:
                imgs1.append(img_space)
            imgs1.append(img_char)

            # for image2: always put char then space
            imgs2.append(img_char)
            imgs2.append(self._draw_character("  ", draw1, char_color))

        # compute final width based on image1’s total text width
        total_text_width = sum(im.size[0] for im in imgs1)
        total_text_width_2 = sum(im.size[0] for im in imgs2)

        # compose both images
        img = self._compose_image(base1, imgs1, total_text_width, len(chars))
        clean = self._compose_image(base2, imgs2, total_text_width_2, len(chars), offset=False)

        return img, clean

    def generate_image(self, chars: str,
                       bg_color: ColorTuple | None = None,
                       fg_color: ColorTuple | None = None) -> Image:
        """Generate the image of the given characters.

        :param chars: text to be generated.
        :param bg_color: background color of the image in rgb format (r, g, b).
        :param fg_color: foreground color of the text in rgba format (r,g,b,a). This will be the default, but each character gets a random color.
        """
        background = bg_color if bg_color else random_color(238, 255)
        # Default foreground color, though it will be overridden per character
        default_fg_color = fg_color if fg_color else random_color(10, 200, secrets.randbelow(36) + 220)

        im, clean = self.create_captcha_image(chars, default_fg_color, background)
            
        # The noise color is now independent of a single text color
        if self._difficulty in ('hard', 'bonus'):
            for _ in range(random.randint(1, 2)):
                noise_color = random_color(10, 200, secrets.randbelow(36) + 220)        
                self.create_noise_curve(im, noise_color)
            self.create_noise_dots(im, noise_color)
        
        im = im.filter(SMOOTH)
        clean = clean.filter(SMOOTH)
        return im, clean


    def generate(self, chars: str, format: str = 'png',
                 bg_color: ColorTuple | None = None,
                 fg_color: ColorTuple | None = None) -> BytesIO:
        """Generate an Image Captcha of the given characters.

        :param chars: text to be generated.
        :param format: image file format
        :param bg_color: background color of the image in rgb format (r, g, b).
        :param fg_color: foreground color of the text in rgba format (r,g,b,a).
        """
        im, clean = self.generate_image(chars, bg_color=bg_color, fg_color=fg_color)
        out = BytesIO()
        im.save(out, format=format)
        out.seek(0)
        return out


    def write(self, chars: str, output: str, output_clean, format: str = 'png',
              bg_color: ColorTuple | None = None,
              fg_color: ColorTuple | None = None) -> None:
        """Generate and write an image CAPTCHA data to the output.

        :param chars: text to be generated.
        :param output: output destination.
        :param format: image file format
        :param bg_color: background color of the image in rgb format (r, g, b).
        :param fg_color: foreground color of the text in rgba format (r,g,b,a).
        """
        im, clean = self.generate_image(chars, bg_color=bg_color, fg_color=fg_color)
        im.save(output, format=format)
        clean.save(output_clean, format=format)
    
    
    def write_clean(self, text: str, output: str, TEXT_HEIGHT = 50) -> None:
        """Renders a clean, centered version of the text."""
        canvas = createImage("RGB", (self._width, self._height), "white")
        font = ImageFont.truetype(str(self._fonts[0]), TEXT_HEIGHT)
        # Use getbbox for accurate centering
        spaced = " ".join(text)
        l, t, r, b = font.getbbox(spaced)
        w, h = r - l, b - t
        left = (self._width - w) // 2
        top  = (self._height - h) // 2
        Draw(canvas).text((left - l, top - t), spaced, font=font, fill="black")
        # Write to path
        canvas.save(output)



# Outside class
def random_color(
        start: int,
        end: int,
        opacity: int | None = None) -> ColorTuple:
    red = secrets.randbelow(end - start + 1) + start
    green = secrets.randbelow(end - start + 1) + start
    blue = secrets.randbelow(end - start + 1) + start
    if opacity is None:
        return red, green, blue
    return red, green, blue, opacity



# Example of how to use the updated class:
if __name__ == '__main__':
    
    # Fonts
    FONT_PATHS = [
        "/home/user/ocr/task0_dataset/fonts/OpenSans_Condensed-Regular.ttf",
        "/home/user/ocr/task0_dataset/fonts/BebasNeue-Regular.ttf",
        "/home/user/ocr/task0_dataset/fonts/Royalacid.ttf",
    ]
    
    # RESOLUTION
    WIDTH = 256
    HEIGHT = 256
    
    easy = ImageCaptcha(width = WIDTH, height = HEIGHT,
                    fonts=FONT_PATHS, font_sizes = (62, 70, 76)
    )
    medium = ImageCaptcha(
        # resolution
        width  = WIDTH,
        height = HEIGHT,
        # distortion
        character_offset_dx = (0, 4),
        character_offset_dy = (0, 6),
        character_rotate  = (-20, 20),
        character_warp_dx = (0.1, 0.3),
        character_warp_dy = (0.2, 0.3),
        word_space_probability =  0.35,
        word_offset_dx = 0.35,
        # fonts
        fonts      = FONT_PATHS, 
        font_sizes = (40, 48, 55),
        # difficulty
        difficulty = "medium", 
        max_gaussian_noise_rate=0.2
    )
    hard = ImageCaptcha(
        # resolution
        width  = WIDTH,
        height = HEIGHT,
        # distortion
        character_offset_dx = (1, 6),
        character_offset_dy = (1, 7),
        character_rotate  = (-30, 30),
        character_warp_dx = (0.2, 0.4),
        character_warp_dy = (0.3, 0.4),
        word_space_probability =  0.55,
        word_offset_dx = 0.45,
        # fonts
        fonts      = FONT_PATHS, 
        font_sizes = (40, 48, 55),
        # difficulty
        difficulty = "hard", 
        max_gaussian_noise_rate=0.4
    )

    # The text to generate
    text = 'IUHI7' # 3 min -> 6 max 

    # Write the image to a file
    hard.write(text, output=f'task0_dataset/samples/hard-{text}.png', output_clean=f'task0_dataset/samples/CLEAN-{text}.png')

    print(f"\nGenerated '{text}' captchas\n")