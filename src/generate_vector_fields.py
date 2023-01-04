import numpy as np
import colorsys
from PIL import Image
from multiprocessing import Pool


def color_pixel_circle(bm_pxl, width, height, angle_offset, dots, r=0.25):
    # print(width, height, angle_offset, r)
    # print_b = bm_pxl[0] == 75 and bm_pxl[1] == 0 and False
    wh = np.asarray([width / 2., height / 2.])

    clockwise = dots[:, -1]
    coordinates = dots[:, :-1]

    xy = (bm_pxl[:-1] - wh) / wh

    diff = xy - coordinates
    diff_l = np.linalg.norm(diff, axis=-1, keepdims=True)

    dist = (np.sum(diff ** 2, axis=-1) ** .5 - r) / r * 1
    dist_r = (1 - np.exp(-np.abs(dist)))  # *5 ))
    dist_w = np.exp(-diff_l * 20)
    # if print_b:
    #     print(dist, 'dist')
    #     print(dist_w, ' dist_w')
    #     print(dist_r, 'dist_r')

    vec = diff / np.maximum(diff_l, 1.0e-36) * dist_w
    # if print_b:
    #     print(vec, 'vec normed & weighted')
    vec = (vec * np.asarray([[1, -1]]))[:, ::-1]
    # if print_b:
    #     print(vec, 'vec transformed')
    vec = vec * np.expand_dims((-1) ** (clockwise == 0), axis=-1)
    # if print_b:
    #     print(vec, 'vec clw')

    dist_ = ((-1) ** np.logical_xor(clockwise, dist < 0)) * dist_r * np.pi / 4
    _sin = np.sin(dist_)
    _cos = np.cos(dist_)
    vec_x = np.sum(vec[:, 0] * _cos + vec[:, 1] * _sin)
    vec_y = np.sum(vec[:, 0] * (-_sin) + vec[:, 1] * _cos)
    # vec_x = np.sum(vec[:, 0])
    # vec_y = np.sum(vec[:, 1])

    # if print_b:
    #     print(dist_, 'dist_')
    default_angle = (np.arctan2(vec_y, vec_x)) - np.radians(angle_offset)  # - np.pi / 2

    # vec = np.sum(vec, axis=0)
    # if print_b:
    #     print(vec, 'vec sum')
    # default_angle = (np.arctan2(vec[1], vec[0])) #+ (np.pi/2)

    value = np.sum(default_angle) % (2 * np.pi)
    # value = np.sum(default_angle) / (2 * np.pi)
    # value = value - value // 1
    # value = 1 - value if value < 0 else value

    return colorsys.hsv_to_rgb(value / (2 * np.pi), 1, 0.8)

    # dist = (((x_ ** 2 + y_ ** 2) ** .5 - r) / r * 2)
    # dist = ((-1) ** int(dist < 0)) * (1 - np.exp(- abs(dist)))
    # add_degree = dist * (np.pi / 8) + np.radians(angle_offset) + np.pi / 4
    # value = (np.arctan2(y_, x_) + add_degree + 4 * np.pi) / (2 * np.pi)
    # value = value - value // 1
    # return colorsys.hsv_to_rgb(value, 1, 0.6 + abs(dist) * .3)[::-1]
    # return colorsys.hsv_to_rgb(value, 1, 0.8)[::-1]
    # return np.asarray([dist, add_degree, value])


def color_pixel_square(bm_pxl, width, height, left, r=0.5):
    wh = np.asarray([width / 2., height / 2.])
    xy = (bm_pxl[:-1] - wh) / wh
    dist = np.abs(xy) - r
    idx_xy = np.argmin(np.abs(dist))
    if np.abs(dist[idx_xy]) < 0.05:
        angle = [-0.5*np.pi, 0][idx_xy] + [0, np.pi][xy[idx_xy] < 0]
        color = colorsys.hsv_to_rgb((angle % (2 * np.pi)) / (2 * np.pi), 1, 0.8)
        return color
    else:
        angle = 2*np.pi + [0, 0.5*np.pi][idx_xy] + [np.pi, 0][(dist[idx_xy]*(-1)**(xy[idx_xy]<0)) < 0]
        color = colorsys.hsv_to_rgb((angle % (2 * np.pi)) / (2 * np.pi), 1, 0.8)
        return color


def make_image_circle(args):
    width, height, angle_offset, dots, r = args
    print('begin %f' % angle_offset)
    bm = np.asarray([[(x, y, 0) for x in range(width)] for y in range(height)])
    bm = np.apply_along_axis(color_pixel_circle, -1, bm,
                             width=width, height=height,
                             angle_offset=angle_offset, dots=dots, r=r)
    bm = (bm * 255).astype(np.uint8)
    im = Image.fromarray(bm)
    im.save('Vectorfield_spiral_r_%f_angleOffset_%f_500x500.png' % (r, angle_offset))
    print('done %s' % angle_offset)
    return None


def make_image_square(args):
    r, width, height, file_name = args
    bm = np.asarray([[(x, y, 0) for x in range(width)] for y in range(height)])
    bm = np.apply_along_axis(color_pixel_square, -1, bm,
                             width=width, height=height,
                             left=True, r=r)
    print(np.max(bm))
    bm = (bm * 255).astype(np.uint8)
    im = Image.fromarray(bm)
    im.save(file_name)
    print('done %s' % file_name)
    return None


def generate_circless():
    # r = 250.0
    r = np.asarray([.30, .30, .30, .30,
                    .25, .25, .25, .25,
                    .3, .3, .3, .3,
                    .3, .3, .3, .3,
                    .3, .3, .3, .3,
                    ])
    r = 0.27
    # r = 1
    width = 500
    height = 500
    dots = np.asarray([
        [0.5, 0, True], [-0.5, 0, True], [0, 0.5, True], [0, -0.5, True],
        [0.4, 0.4, False], [0.4, -0.4, False], [-0.4, 0.4, False], [-0.4, -0.4, False],
        # inner drill
        [0.2, 0.2, True], [0.2, -0.2, True], [-0.2, 0.2, True], [-0.2, -0.2, True],
        # outer drill
        [0.7, 0.2, False], [0.7, -0.2, False], [-0.7, 0.2, False], [-0.7, -0.2, False],
        [0.2, 0.7, False], [-0.2, 0.7, False], [0.2, -0.7, False], [-0.2, -0.7, False],
    ])
    dots = np.asarray([
        # [0, 0.5, False],
        # [0, -0.5, True],
        [0, 0, True]
    ])

    pool = Pool(processes=16)
    pool.map(make_image_circle, [(width, height, angle, dots, r) for angle in [0,
                                                                               # 5, -5, 10, -10, 20, -20, 30, -30
                                                                               ]
                                 ])
    # for angle_offset in [0,]:#5,-5,10,-10,20,-20,30,-30]:
    #     print(angle_offset)
    #     # bm = np.zeros((width, height, 3))
    #     bm = np.asarray([[(x, y, 0) for x in range(width)] for y in range(height)])
    #     bm = np.apply_along_axis(color_pixel, -1, bm,
    #                              width=width, height=height,
    #                              angle_offset=angle_offset, dots=dots, r=r)
    #     # print(np.min(np.abs(bm[:, :, 0])))
    #     # print(np.min(bm[:, :, 0]))
    #     # print(np.max(bm[:, :, 0]))
    #     # print()
    #     # print(bm[:,:,1][np.unravel_index(np.argmin(np.abs(bm[:,:,0])), bm.shape[:-1])])
    #     # print(bm[:,:,1][np.unravel_index(np.argmin(bm[:,:,0]), bm.shape[:-1])])
    #     # print(bm[:,:,1][np.unravel_index(np.argmax(bm[:,:,0]), bm.shape[:-1])])
    #
    #     bm = (bm * 255).astype(np.uint8)
    #     im = Image.fromarray(bm)
    #     im.save('Vectorfield_r_%f_angleOffset_%f.png' % (r, angle_offset))


if __name__ == '__main__':
    r = 0.5
    width = 500
    height = 500
    file_name = 'Square_r%0.02f.png' % r
    make_image_square((r, width, height, file_name))
