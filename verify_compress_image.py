from PIL import Image


def main():
    img = Image.open('test1.jpg')
    img = img.resize((640, 480), Image.ANTIALIAS)
    img.save('test1_compressed.jpg', Optimize=True)


main()