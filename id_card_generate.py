import PIL.Image as PImage
import cv2
import numpy
from PIL import ImageFont, ImageDraw
from config import *


def change_background(img, img_back, zoom_size, center):
    # 缩放
    img = cv2.resize(img, zoom_size)
    rows, cols, channels = img.shape

    # 转换hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 获取mask
    # lower_blue = np.array([78, 43, 46])
    # upper_blue = np.array([110, 255, 255])
    diff = [5, 30, 30]
    gb = hsv[0, 0]
    lower_blue = numpy.array(gb - diff)
    upper_blue = numpy.array(gb + diff)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # cv2.imshow('Mask', mask)

    # 腐蚀膨胀
    erode = cv2.erode(mask, None, iterations=1)
    dilate = cv2.dilate(erode, None, iterations=1)

    # 粘贴
    for i in range(rows):
        for j in range(cols):
            if dilate[i, j] == 0:  # 0代表黑色的点
                img_back[center[0] + i, center[1] + j] = img[
                    i, j
                ]  # 此处替换颜色，为BGR通道

    return img_back


def paste(avatar, bg, zoom_size, center):
    avatar = cv2.resize(avatar, zoom_size)
    rows, cols, channels = avatar.shape
    for i in range(rows):
        for j in range(cols):
            bg[center[0] + i, center[1] + j] = avatar[i, j]
    return bg


# 获得显示长度，中文占2位
def get_show_len(txt):
    len_txt = len(txt)
    len_txt_utf8 = len(txt.encode("utf-8"))
    # 中文字符多算1位
    size = int((len_txt_utf8 - len_txt) / 2 + len_txt)
    return size


# 获得要显示的字符串
# start 显示长度起始位置
# end 显示长度结束位置
def get_show_txt(txt, show_start, show_end):
    def get_show_index(txt, i_len):
        res_txt = ""
        for index, char in enumerate(txt):
            res_txt = res_txt + char
            res_show_len = get_show_len(res_txt)
            if res_show_len > i_len:
                return index
        return get_show_len(txt)

    i_start = get_show_index(txt, show_start)
    i_end = get_show_index(txt, show_end)

    return txt[i_start:i_end]


class IDGen:
    def __init__(
        self, name, sex, nation, year, month, day, avatar, addr, idn, org, life, cutout
    ):
        self.eName = name  # 姓名
        self.eSex = sex  # 性别
        self.eNation = nation  # 民族
        self.eYear = year  # 出生年
        self.eMon = month  # 月
        self.eDay = day  # 日
        self.eAvatar = avatar  # 头像
        self.eAddr = addr  # 住址
        self.eIdn = idn  # 证件号码
        self.eOrg = org  # 签发机关
        self.eLife = life  # 有效期限
        self.cutout = cutout

    def handle_image(self):
        avatar = PImage.open(self.eAvatar)
        # 500x670
        empty_image = PImage.open("asserts/empty.png")

        name_font = ImageFont.truetype("asserts/fonts/hei.ttf", 72)
        other_font = ImageFont.truetype("asserts/fonts/hei.ttf", 64)
        birth_date_font = ImageFont.truetype("asserts/fonts/fzhei.ttf", 60)
        id_font = ImageFont.truetype("asserts/fonts/ocrb10bt.ttf", 90)

        draw = ImageDraw.Draw(empty_image)
        draw.text((630, 690), self.eName, fill=(0, 0, 0), font=name_font)
        draw.text((630, 840), self.eSex, fill=(0, 0, 0), font=other_font)
        draw.text((1030, 840), self.eNation, fill=(0, 0, 0), font=other_font)
        draw.text((630, 975), self.eYear, fill=(0, 0, 0), font=birth_date_font)
        draw.text((950, 975), self.eMon, fill=(0, 0, 0), font=birth_date_font)
        draw.text((1150, 975), self.eDay, fill=(0, 0, 0), font=birth_date_font)

        # 住址
        addr_loc_y = 1115
        addr_lines = self.get_addr_lines()
        for addr_line in addr_lines:
            draw.text((630, addr_loc_y), addr_line, fill=(0, 0, 0), font=other_font)
            addr_loc_y += 100

        # 身份证号
        draw.text((900, 1475), self.eIdn, fill=(0, 0, 0), font=id_font)

        # 背面
        draw.text((1050, 2750), self.eOrg, fill=(0, 0, 0), font=other_font)
        draw.text((1050, 2895), self.eLife, fill=(0, 0, 0), font=other_font)

        if self.cutout == True:
            avatar = cv2.cvtColor(numpy.asarray(avatar), cv2.COLOR_RGBA2BGRA)
            empty_image = cv2.cvtColor(numpy.asarray(empty_image), cv2.COLOR_RGBA2BGRA)
            empty_image = change_background(
                avatar, empty_image, (500, 670), (690, 1500)
            )
            empty_image = PImage.fromarray(
                cv2.cvtColor(empty_image, cv2.COLOR_BGRA2RGBA)
            )
        else:
            avatar = avatar.resize((500, 670))
            avatar = avatar.convert("RGBA")
            empty_image.paste(avatar, (1500, 690), mask=avatar)

        empty_image.save("彩色.png")
        empty_image.convert("L").save("黑白.png")

    # 获得要显示的住址数组
    def get_addr_lines(self):
        addr = self.eAddr
        addr_lines = []
        start = 0
        while start < get_show_len(addr):
            show_txt = get_show_txt(addr, start, start + 22)
            addr_lines.append(show_txt)
            start = start + 22

        return addr_lines


if __name__ == "__main__":
    id_gen = IDGen(
        NAME, SEX, NATION, YEAR, MONTH, DAY, AVATAR, ADDR, IDN, ORG, LIFE, False
    )
    id_gen.handle_image()
    print("done")
