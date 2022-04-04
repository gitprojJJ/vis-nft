from base64 import encode
import base64
import pydenticon
import hashlib


foreground = [
    "rgb(45,79,255)",
    "rgb(254,180,44)",
    "rgb(226,121,234)",
    "rgb(30,179,253)",
    "rgb(232,77,65)",
    "rgb(49,203,115)",
    "rgb(141,69,170)"
]

generator = pydenticon.Generator(
    rows=7,
    columns=7,
    digest=hashlib.sha1,
    foreground=foreground,
    background="rgb(224,224,224)"
)

def address_to_png(address, height=30, width=30):
    return generator.generate(
        address,
        height=height,
        width=width,
        output_format="png"
    )

def address_to_md(address, height=30, width=30):
    encoded_image = base64.b64encode(
        address_to_png(address, height, width)
    ).decode("UTF-8")
    return f"<img src='data:image/png;base64,{encoded_image}'/>"