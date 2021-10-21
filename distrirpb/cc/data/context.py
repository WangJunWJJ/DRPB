# -*- coding: utf-8 -*-
# @Time : 2021/3/26 上午10:02
# @Author : Xiaochuan Zhang


context = None

#define the list context

class Context(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getattr__(self, key):
        return self.__dict__[key]


def _context():
    global context
    if context is None:
        context = Context(backend='torch', mode='deploy')
    return context


def set_context(**kwargs):
    ctx = _context()
    for k, v in kwargs.items():
        setattr(ctx, k, v)


def get_context(key):
    ctx = _context()
    return getattr(ctx, key)
