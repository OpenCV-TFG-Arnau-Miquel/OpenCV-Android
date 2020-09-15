package com.example.androidtfg;

import org.opencv.core.Point;
import org.opencv.core.Rect;

class Detection {
    private Rect box;
    private int id;
    private int intConf;
    private Point tl;

    public Detection(Rect box, int id, int intConf, Point tl) {
        this.box = box;
        this.id = id;
        this.intConf = intConf;
        this.tl = tl;
    }

    public Rect getBox() {
        return box;
    }

    public int getId() {
        return id;
    }

    public int getIntConf() {
        return intConf;
    }

    public Point getTl() {
        return tl;
    }
}
