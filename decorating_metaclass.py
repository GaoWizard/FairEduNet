from abc import ABCMeta


# 定义判断方法是否需要装饰的函数
def do_decorate(attr, value):
    return not attr.startswith('_') and callable(value) and getattr(value, '__decorate__', True)

# 工程函数，生成元类.
def factory(decorator):
    class ApplyDecoratorMeta(ABCMeta):
        """
        元类(Metaclass)，它能对所有公有的、非特殊的实例方法应用一个装饰器.

        注意:
            `decorator` m必须使用 @functools.wraps(f) 这种格式，才能使 abstractmethod 正常工作.

        思路来源参考:
        https://stackoverflow.com/questions/10067262/automatically-decorating-every-instance-method-in-a-class
        """
        # 定义一个__new__()方法，以实现自动装饰
        def __new__(cls, name, bases, dct):
            for attr, value in dct.items():
                if do_decorate(attr, value):
                    dct[attr] = decorator(value)
            # 调用父类的__new__方法，传入参数
            return super(ApplyDecoratorMeta, cls).__new__(cls, name, bases, dct)
    return ApplyDecoratorMeta

# 装饰器，用于标记不需要自动装饰的函数
def dont_decorate(func):
    func.__decorate__ = False
    return func

# 定义ApplyDecorator函数，ApplyDecorator函数通过factory和元类创建,返回一个可自动装饰的类.
def ApplyDecorator(decorator):
    # 使用factory和传入的decorator生成了一个元类,然后调用这个元类创建了一个名为'ApplyDecorator'的类
    # 这个类将自动应用decorator进行装饰
    return factory(decorator)(str('ApplyDecorator'), (), {})
