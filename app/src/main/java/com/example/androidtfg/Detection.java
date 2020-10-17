package com.example.androidtfg;

import org.opencv.core.Point;
import org.opencv.core.Rect;

class Detection {
    private Rect box;
    private int id;
    private int intConf;
    private Point textPosition;

    public Detection(Rect box, int id, int intConf, Point textPosition) {
        this.box = box;
        this.id = id;
        this.intConf = intConf;
        this.textPosition = textPosition;
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

    public Point getTextPosition() {
        return textPosition;
    }
}
