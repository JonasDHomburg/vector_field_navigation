#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QMainWindow, QDesktopWidget, QApplication, QLabel, \
    QSpacerItem, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np
import colorsys

import sys


class MainWidget(QWidget):
    def __init__(self, main_window):
        super(MainWidget, self).__init__(main_window)
        self.pixmap_label = QLabel(main_window)
        width = 1000
        height = 1000
        self.im_np = np.zeros((height, width, 3), dtype=np.uint8)
        pixels_y = np.arange(0, self.im_np.shape[0])
        pixels_x = np.arange(0, self.im_np.shape[1])
        pixels_x, pixels_y = np.meshgrid(pixels_x, pixels_y)
        self._pixels_x = pixels_x.reshape((-1, 1))
        self._pixels_y = pixels_y.reshape((-1, 1))
        # self.points = None
        self.points = np.asarray([
            [int(0.2 * height), int(0.2 * width)],
            [int(0.2 * height), int(0.8 * width)],
            [int(0.8 * height), int(0.8 * width)],
            # [int(0.8 * height), int(0.2 * width)],
            # [int(0.2 * height), int(0.2 * width)],
        ], dtype=np.int).T
        print(self.points)

        self.create_view()
        self.connect_events()
        self.update_image()
        self.update_ndarray2()

    def create_view(self):
        self.pixmap_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        self.setLayout(layout)

        layout_ = QVBoxLayout()
        layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Expanding))
        layout.addLayout(layout_)
        layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Expanding))

        layout_.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Expanding))
        layout_.addWidget(self.pixmap_label, alignment=Qt.AlignCenter)
        layout_.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Expanding))

    def connect_events(self):
        self.pixmap_label.mousePressEvent = self.img_mouse_press
        self.pixmap_label.mouseMoveEvent = self.img_mouse_move
        self.pixmap_label.mouseReleaseEvent = self.img_mouse_release

    def event2ndarray(self, event):
        return np.asarray([[
            event.y() / (self.pixmap_label.height() / self.im_np.shape[0]),
            event.x() / (self.pixmap_label.width() / self.im_np.shape[1]),
        ]], dtype=np.int).T

    def img_mouse_press(self, event):
        if self.points is None:
            self.points = self.event2ndarray(event)
        else:
            self.points = np.concatenate([self.points, self.event2ndarray(event)], axis=-1)
        # self.update_ndarray()

    def img_mouse_move(self, event):
        # self.points = np.concatenate([self.points, self.event2ndarray(event)], axis=-1)
        # print(self.points)
        # self.update_ndarray()

        pass

    def img_mouse_release(self, event):
        # self.points = np.concatenate([self.points, self.event2ndarray(event)], axis=-1)
        self.update_ndarray2()
        print(self.points)
        pass

    @staticmethod
    def hsv_to_rgb(hsv):
        rgb = np.zeros(hsv.shape, dtype=np.float)
        s_eq_0 = np.where(hsv[:, :, 1] == 0.0)
        rgb[s_eq_0] = np.expand_dims(hsv[:, :, 2][np.where(hsv[:, :, 1] == 0.0)], axis=-1)
        s_neq_0 = np.where(hsv[:, :, 1] != 0.0)

        h = hsv[:, :, 0][s_neq_0]
        s = hsv[:, :, 1][s_neq_0]
        v = hsv[:, :, 2][s_neq_0]
        i = (h * 6.0).astype(np.int)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6

        i_idx = np.where(i == 0)
        rgb[s_neq_0[0][i_idx], s_neq_0[1][i_idx], 0] = v[i_idx]
        rgb[s_neq_0[0][i_idx], s_neq_0[1][i_idx], 1] = t[i_idx]
        rgb[s_neq_0[0][i_idx], s_neq_0[1][i_idx], 2] = p[i_idx]
        i_idx = np.where(i == 1)
        rgb[s_neq_0[0][i_idx], s_neq_0[1][i_idx], 0] = q[i_idx]
        rgb[s_neq_0[0][i_idx], s_neq_0[1][i_idx], 1] = v[i_idx]
        rgb[s_neq_0[0][i_idx], s_neq_0[1][i_idx], 2] = p[i_idx]
        i_idx = np.where(i == 2)
        rgb[s_neq_0[0][i_idx], s_neq_0[1][i_idx], 0] = p[i_idx]
        rgb[s_neq_0[0][i_idx], s_neq_0[1][i_idx], 1] = v[i_idx]
        rgb[s_neq_0[0][i_idx], s_neq_0[1][i_idx], 2] = t[i_idx]
        i_idx = np.where(i == 3)
        rgb[s_neq_0[0][i_idx], s_neq_0[1][i_idx], 0] = p[i_idx]
        rgb[s_neq_0[0][i_idx], s_neq_0[1][i_idx], 1] = q[i_idx]
        rgb[s_neq_0[0][i_idx], s_neq_0[1][i_idx], 2] = v[i_idx]
        i_idx = np.where(i == 4)
        rgb[s_neq_0[0][i_idx], s_neq_0[1][i_idx], 0] = t[i_idx]
        rgb[s_neq_0[0][i_idx], s_neq_0[1][i_idx], 1] = p[i_idx]
        rgb[s_neq_0[0][i_idx], s_neq_0[1][i_idx], 2] = v[i_idx]
        i_idx = np.where(i == 5)
        rgb[s_neq_0[0][i_idx], s_neq_0[1][i_idx], 0] = v[i_idx]
        rgb[s_neq_0[0][i_idx], s_neq_0[1][i_idx], 1] = p[i_idx]
        rgb[s_neq_0[0][i_idx], s_neq_0[1][i_idx], 2] = q[i_idx]
        return rgb

    def update_ndarray(self):
        if self.points.shape[1] < 2:
            return
        # im_vec = np.zeros((self.im_np.shape[0], self.im_np.shape[1], 2), dtype=np.float)
        points_ = self.points.astype(np.float)
        p_vec = np.concatenate([
            points_[:, (1,)] - points_[:, (0,)],
            points_[:, 2:] - points_[:, :-2],
            points_[:, (-1,)] - points_[:, (-2,)],
        ], axis=-1).astype(np.float)

        angles = np.arctan2(p_vec[1, :], p_vec[0, :]) % (2 * np.pi)
        rgb = (self.hsv_to_rgb(np.expand_dims(np.stack([
            angles,
            np.ones(angles.shape) * 1.0,
            np.ones(angles.shape) * 0.8,
        ], axis=-1), axis=1))[:, 0, :] * 255).astype(np.uint8)
        self.im_np[self.points[0, :], self.points[1, :], :] = rgb
        self.update_image()

    def update_ndarray2(self):
        if self.points.shape[1] < 2:
            return
        P2F = self.points.astype(np.float)
        # xs, ys = self._pixels_x[:50], self._pixels_y[:50]
        xs, ys = self._pixels_x, self._pixels_y

        vx = P2F[1, 1:] - P2F[1, :-1]
        vy = P2F[0, 1:] - P2F[0, :-1]
        vxy_norm = np.sqrt(vx ** 2 + vy ** 2)
        vx_ = vx / vxy_norm
        vy_ = vy / vxy_norm

        acx = P2F[1, :-1] - xs
        acy = P2F[0, :-1] - ys

        t = (acy * vx - acx * vy) / vxy_norm

        t_ = np.argmin(np.abs(t), axis=-1)
        # t = t[(np.arange(t.shape[0]), t_)]
        # t = np.expand_dims(t, axis=-1)

        r = (-acx * vx_ - acy * vy_) / vxy_norm

        angle = (1 - np.exp(-np.abs(t) / 50)) * np.sign(t) * (-np.pi / 2)
        print(angle.shape)
        sin_ = np.sin(angle)
        cos_ = np.cos(angle)

        p_vec_x = (vx_ * cos_ + vy_ * sin_)
        p_vec_y = (vx_ * (-sin_) + vy_ * cos_)

        print(np.argpartition(np.abs(t), kth=1, axis=-1).shape)
        print(np.argpartition(np.abs(t), kth=1, axis=-1)[0, 0])
        print(np.argpartition(np.abs(t), kth=1, axis=-1)[0, 1])
        t_ = np.argpartition(np.abs(t), kth=1, axis=-1)
        r_ = np.logical_and(r >= 0, r <= 1)

        p_vec_x = p_vec_x[np.arange(t.shape[0]), t_[:, 0]] * r_[np.arange(t.shape[0]), t_[:, 0]] * (1 / (np.abs(t[np.arange(t.shape[0]), t_[:, 0]]) + 1e-12) * 10) + \
                  p_vec_x[np.arange(t.shape[0]), t_[:, 1]] * r_[np.arange(t.shape[0]), t_[:, 1]] * (1 / (np.abs(t[np.arange(t.shape[0]), t_[:, 1]]) + 1e-12) * 10)
        p_vec_y = p_vec_y[np.arange(t.shape[0]), t_[:, 0]] * r_[np.arange(t.shape[0]), t_[:, 0]] * (1 / (np.abs(t[np.arange(t.shape[0]), t_[:, 0]]) + 1e-12) * 10) + \
                  p_vec_y[np.arange(t.shape[0]), t_[:, 1]] * r_[np.arange(t.shape[0]), t_[:, 1]] * (1 / (np.abs(t[np.arange(t.shape[0]), t_[:, 1]]) + 1e-12) * 10)

        print(np.logical_and(r >= 0, r <= 1).shape)
        print((1.1 - np.exp(-np.abs(t) / 1000)).shape)

        # p_vec_x = np.sum(p_vec_x * np.logical_and(r >= 0, r <= 1) * (1 / (np.abs(t) + 1e-12) * 10), axis=-1)
        # p_vec_y = np.sum(p_vec_y * np.logical_and(r >= 0, r <= 1) * (1 / (np.abs(t) + 1e-12) * 10), axis=-1)

        # p_vec_x = p_vec_x[np.arange(t.shape[0]), t_]
        # p_vec_y = p_vec_y[np.arange(t.shape[0]), t_]

        rgb = (self.hsv_to_rgb(np.expand_dims(np.stack([
            (np.arctan2(p_vec_y, p_vec_x) % (2 * np.pi)) / (2 * np.pi),
            # (np.sum(np.logical_and(r >= 0, r <= 1), axis=-1) >= 1) * 0.5,
            np.ones(p_vec_x.shape) * 1.0,
            np.ones(p_vec_x.shape) * 0.8,
        ], axis=-1), axis=1))[:, :, :] * 255).astype(np.uint8)
        self.im_np[ys, xs, :] = rgb
        self.update_image()

    def update_image(self):
        qimage = QImage(self.im_np.data, self.im_np.shape[1], self.im_np.shape[0],
                        3 * self.im_np.shape[1], QImage.Format_RGB888)
        pixmap = QPixmap(qimage)
        pixmap = pixmap.scaled(self.im_np.shape[1], self.im_np.shape[0], Qt.KeepAspectRatio)
        self.pixmap_label.setPixmap(pixmap)


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.resize(1000, 1000)
        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        self.setWindowTitle("VectorFieldDrawer")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    app.setActiveWindow(win)
    win.setCentralWidget(MainWidget(win))
    win.show()
    sys.exit(app.exec_())
