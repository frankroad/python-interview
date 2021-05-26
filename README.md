### Python 面试题 

#### 1 命名空间(Namespace) 

```python

>>> a = 5
>>> globals()
{'a': 5, ...}  a 指向5这个对象 

>>> a = "test"
>>> globals()
{'a': 'test', ...}  a 指向'test'这个对象

在Python源码中, 有这样一句话: Names have no type, but objects do.

Python的名字实际上是一个字符串对象,

它和所指向的目标对象一起在名字空间中构成一项 {name: object} 关联。

这样Python就拥有了动态语言的动能

```

#### 2 传参(Pass A Variable) 

```python

当一个参数传递给函数的时候, 相当于a (local) = a (global) 赋值操作

虽然它们不是同一个变量, 但它们任然指向同一个对象.

# immutable  For example string, tuple and number
a = 1

print(id(a))  # 10914368


def fun(a):
    print(id(a))  # 10914368
    a = 2         #　这里改变a (local) 指向的对象
    print(id(a))  # 10914400


fun(a)
print(id(a))  # 10914368

# mutable  For example list, dict and set
a = [] 
print(id(a))  # 140408380111368


def fun(a):
    print(id(a))  # 140408380111368
    a.append(1)   # 这里没有改变a (local) 指向的对象
    print(id(a))  # 140408380111368


fun(a)
print(id(a))  # 140408380111368

```

#### 3 实例变量和类变量(instance Variable and class Variable)

```python

class Person:
    name = "aaa"


p1 = Person()
p2 = Person()
p1.name = "bbb"         # 创建实例变量name 并指向新的对象
print(id(p1.name))      # 140078039596032
print(id(p2.name))      # 140078039563040 访问类变量
print(id(Person.name))  # 140078039563040 访问类变量


class Person:
    name = [] 


p1 = Person()
p2 = Person()
p1.name.append(1)       # 访问类变量
print(id(p1.name))      # 140078039616392 访问类变量
print(id(p2.name))      # 140078039616392 访问类变量
print(id(Person.name))  # 140078039616392 访问类变量

```

#### 4 列表推导(List Comprehension) 

```python

a = [1, 2, 3, 4]

[x**2 for x in a]  # [1, 4, 9. 16]


表达式会按照从左至右的顺序来执行

matrix= [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

[x for row in matrix for x in row] 
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

[[x**2 for x in row] for row in matrix]
# [[1, 4, 9], [16, 25, 36], [49, 64, 81]]

[[x for x in row if x % 3 ==0] for row in matrix if sum(row) >= 10]


这个例子已经有点复杂。

在列表推导中，最好不要使用两个以上的表达式。
可以使用两个条件、两个循环或一个条件搭配一个循环。

如果很复杂，应该用 if 和 for 语句写成辅助函数。

matrix= [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

[[x for x in row if x % 3 == 0] for row in matrix if sum(row) >= 10]
# [[6], [9]]


```

#### 函数式编程代替列表推导 

```python

a = [1, 2, 3, 4]

b = map(lambda x: x**2, a)  # 惰性生成
list(b)  # [1, 4, 9, 16]

a = [1, 2, 3, 4]

b = filter(lambda x: x % 2 == 0, a)  # 惰性生成
list(b)  # [2, 4]

```


#### 5 生成器(Generator) 

```python

通常生成器是通过调用一个或多个yield表达式构成的函数生成的

def t(): 
    yield 5


t()  # <generator object ...>

l = [x for x range(5)]  # [0, 1, 2, 3, 4]

g = (x for x in range(5))  # <generator object ...>

注意：
    由生成器表达式所返回的那个迭代器是有状态的，
	用过一轮之后，就不要反复使用了

g = (x for x in range(5))
print(list(g))  # [0, 1, 2, 3, 4]
print(list(g))  # []

对于列表生成式，当输入的数据比较少时，不会出问题。

但如果输入的数据非常多，那么可能会消耗大量内存，并导致程序崩溃。

这时候考虑生成器。

g = (len(x) for x in open('t.txt'))

```

#### 6 字典推导(Dict Comprehension) 

```python

keys = ["name", "age"]
values = ["Tom", 18]
d = {key: value for(key, value) in zip(keys, values)}
print(d)

# {'name': 'Tom', 'age': 18}

```

#### 7 单个下划线和两个下划线(Single Underscore and Double Underscore) 

```python

class T(object):

    def __init__(self, name, age):
        self._name = name 
        self.__age = age


t = T("Tom", 18)
print(t._name)
print(t._T__age)
print(t.__age)

__foo__: 一种约定, Python内部的名字,用来区别其他用户自定义的命名,以防冲突.

_foo: 一种约定,用来指定变量私有. 可以被继承.

__foo: 是真正的私有变量, 不能被继承, 用来区别其它类相同的命名, 
       但也可以通过_classname__foo来访问,
       we are all consenting adults here 
       一般不会通过这种方法来操作变量.

```

#### 8 字符串格式化(format) 

```python

name = "John"
print("I'm {0}".format(name))
# I'm John
print("I'm {name}".format(name=name))
# I'm John
names = ["John", "Tom"]
print("I'm {0[0]}".format(names))
# I'm John

# ^、<、>分别是居中、左对齐、右对齐，后面带宽度
# :号后面带填充的字符，只能是一个字符，不指定的话默认是用空格填充

num = 3
print("{0:0^5}".format(num))
# 00300
print("{0:0<5}".format(num))
# 30000
print("{0:0>5}".format(num))
# 00003
print('{:,}'.format(1234567890))
# 1,234,567,890

```

#### 9 args and kwargs 

```python

它们的顺序必须是

* 位置参数 > 默认参数 > 可变参数 >  命名关键字参数 > 关键字参数


```


#### 位置参数

```python

def t(name):    # name 就是一个位置参数
    print(name)

```


#### 默认参数

```python

def t(name, age=18):  # age 就是一个默认参数
    print(name, age)

t("John")     # John 18
t("Tom", 16)  # Tom 16


默认参数最好用不可变类型


# 在每个人加一个成绩

def t(name, scores=[]):
    scores.append(90)
    print(name, scores)

t("John", [80])
t("Tom", [76])

t("Marsh")
t("Jim")

# John [80, 90]
# Tom [76, 90]
# Marsh [90]
# Jim [90, 90]  # 出现错误



由于 scores 参数的默认值只会在模块加载时计算一次，
所以凡是以默认形式来调用 t 函数的代码，都将共享同一份数组。
这会引发非常奇怪的行为。

def t(name, scores=None):
    if scores is None:
        scores = [90]
    else:
        scores.append(90)
    print(name, scores)

t("John", [80])
t("Tom", [76])

t("Marsh")
t("Jim")

# John [80, 90]
# Tom [76, 90]
# Marsh [90]
# Jim [90]

```
 

#### 可变参数

```python

def t(name, age=18, *args):  # 把 *args 称为可变参数
    print(name, age)         # 会把 78 83 95 转成一个 tuple (元组)
    for score in args:       # 所以当参数特别多时，会耗尽内存，导致崩溃
        print(score)

t("John", 18, 78, 83)        # 这里必须写 age，不然会被后面的值覆盖

或者

scores = [78, 83]
t("John", 18, *scores)

# John 18
# 78
# 83

```


#### 命名关键字参数

```python

必须需要一个或多个关键字参数

def t(name, age=18, *, country, **kw):
    print(name, age)
    print(country)
    for key, value in kw.items():
        print(key, value)

t("John", 18, country="China", gender="male")

或者

def t(name, age=18, *args, country, **kw):
    print(name, age)
    for score in args:
        print(score)
    print(country)
    for key, value in kw.items():
        print(key, value)

t("John", 18, 78, 83, country="China", gender="male")

```


#### 关键字参数

```python

def t(name, age=18, *args, **kw):
    print(name, age)
    for score in args:
        print(score)
    for key, value in kw.items():
        print(key, value)


t("John", 18, 78, 83, gender="male")

或者

others = {"gender": "male"}
t("John", 18, 78, 83, **others)

又或者

scores = [78, 83]
others = {"gender": "male"}
t("John", 18, *scores, **others)

# John 18
# 78
# 83
# gender male

```

#### 10 new and init 

```python

1. __new__是一个静态方法,而__init__是一个实例方法.

2. __new__方法会返回一个创建的实例,而__init__什么都不返回.

3. 只有在__new__返回一个cls的实例时后面的__init__才能被调用.

4. 当创建一个新实例时调用__new__,初始化一个实例时用__init__.

```

#### 11 单例(Singleton) 

```python

在我们工作中经常需要在应用程序中保持一个唯一的实例，

如：IO处理，数据库操作，配置文件等，由于这些对象都要占用重要的系统资源，

所以我们必须始终使用一个公用的实例，如果创造出来多个实例，就会导致许多问题，

比如占用过多资源，不一致的结果等

class Singleton(object):

    def __new__(cls, *args, **kwarg):
        if not hasattr(cls, "_instance"):
            orign = super(Singleton, cls)  # 不清楚什么意思 
            cls._instance = orign.__new__(cls, *args, **kwarg)
        return cls._instance


class MyClass(Singleton):
    pass 


my1 = MyClass()
my2 = MyClass()
print(id(my1))  # 140160695672168
print(id(my2))  # 140160695672168


另外一种单例模式

# mysingleton.py
class My_Singleton(object):
    def foo(self):
            pass

            my_singleton = My_Singleton()

# to use
from mysingleton import my_singleton

my_singleton.foo()

```

#### 12 作用域 

```python

* L （Local） 局部作用域
* E （Enclosing） 闭包函数外的函数中
* G （Global） 全局作用域
* B （Built-in） 内建作用域

以 L --> E --> G --> B 的规则查找，即：在局部找不到，

便会去局部外的局部找，再找不到就会去全局找，再者去内建中找。


__builtins__.a = 0  # Built-in variable
        
b = 1               # Global variable
        

def outside():
    c = 2           # Enclosing variable

    def inside():
        d = 3       # Local variable
        print(a)
        print(b)
        print(c)
        print(d)

    inside()


outside()

# 0
# 1
# 2
# 3


修改Global variable

a = 0              
  
  
def outside():
    b = 1          

    def inside():
        global a    # 与全局变量a指向同一个对象
        a = 100
        c = 2      
        print(a)
        print(b)
        print(c)

    inside()

outside()
print(a)

# 100
# 1
# 2
# 100


修改Enclosing variable

a = 0              
  
  
def outside():
    b = 1          

    def inside():
        nonlocal b    # 与Enclosing variable b指向同一个对象
        b = 10
        c = 2      
        print(a)
        print(b)
        print(c)

    inside()
    print(b)

outside()

# 0
# 10
# 2
# 10


globals

a = 0
print(globals())        # {'a': 0 ...}


def outside():
    globals()['b'] = 1
outside()
print(globals())        # {'a': 0, 'b': 1 ...}


locals

def outside():
    a = 0
    print(locals())     # {'a': 0}
outside()

```

#### 13 浅拷贝和深拷贝(copy and deepcopy) 

```python

import copy
a = [1, 2, 3, 4, ['a', 'b']]  #原始对象

b = a  #赋值，传对象的引用
c = copy.copy(a)  #对象拷贝，浅拷贝
d = copy.deepcopy(a)  #对象拷贝，深拷贝

a.append(5)  #修改对象a
a[4].append('c')  #修改对象a中的['a', 'b']数组对象

print 'a = ', a
print 'b = ', b
print 'c = ', c
print 'd = ', d

输出结果：
a =  [1, 2, 3, 4, ['a', 'b', 'c'], 5]
b =  [1, 2, 3, 4, ['a', 'b', 'c'], 5]
c =  [1, 2, 3, 4, ['a', 'b', 'c']]
d =  [1, 2, 3, 4, ['a', 'b']]

```

#### 14 上下文管理器(contextmanager) 

```python


日志打印 

import logging


def t():
    logging.debug("Some debug data")
    logging.error("Error log here")
    logging.debug("More debug data")


t()

# ERROR:root:Error log here



在上下文环境下日志打印 

import logging
from contextlib import contextmanager
    
    
def t():
    logging.debug("Some debug data")
    logging.error("Error log here")
    logging.debug("More debug data")


@contextmanager
def debug_logging(level):
    logger = logging.getLogger()
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)


with debug_logging(logging.DEBUG):
    print("Inside:")
    t()          # 日志打印的级别为 DEBUG
print("After:")  # 日志打印的级别重新恢复为ERROR
t()


# Inside:
# DEBUG:root:Some debug data
# ERROR:root:Error log here
# DEBUG:root:More debug data
# After:
# ERROR:root:Error log here

```

#### 15 property 

```python


访问属性时，需要表现某种行为时用@property 

class T(object):

    def __init__(self):
        self._value = 5 

    @property                  # 最小惊讶原则
    def value(self):           # 列如
        print("Get value")     # getter里面不应该修改属性
        return self._value     # getter, setter 不应该做复杂或缓慢的操作

    @value.setter
    def value(self, value):
        print("Change value")
        self._value = value


t = T() 
print(t.value)
t.value = 10
print(t.value)

```

#### 16 描述符(descriptor) 

```python

当多个属性都需要表现同一种行为时, 用描述符

from weakref import WeakKeyDictionary


class Grade(object):

    def __init__(self):    
        self._values = WeakKeyDictionary() 

        # 用字典来记录每个实例的状态。
        # 为了避免实例引用计数无法降为0， 
        # 垃圾回收器无法回收，这里用弱引用字典。

    def __get__(self, instance, instance_type):
        if instance is None:
            return self
        return self._values.get(instance, 0)

    def __set__(self, instance, value):
        if not (0 <= value <= 100):
            raise ValueError("Grade must be between 0 and 100")
        self._values[instance] = value


class Exam(object):
    math = Grade()
    english = Grade()


exam = Exam()         
exam.math = 97        # Exam.__dict__['math'].__set__(exam, 40) 
exam.english = 89
print(exam.math)      # Exam.__dict__['math'].__get__(exam, Exam)
print(exam.english)
exam.english = 87
print(exam.english)

# 97
# 89
# 87

```

#### 17 装饰器(decorator) 

```python

装饰器增强函数功能

```


#### 不带参数的装饰器 

```python

def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("Start")
        return func(*args, **kwargs)
    return wrapper


@log             # 相当于 t = log(t)
def t():
    print("End")


t()

# Start
# End

```


#### 带参数的装饰器 

```python

def log(name):
    print(name)
        
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print("Start")
            return func(*args, **kwargs)
        return wrapper
    return decorator


@log("Tom")      # 相当于 t = log("Tom")(t)
def t():
    print("End")


t()

# Tom
# Start
# End

```


#### 同时支持两种方式

```python

from functools import wraps, partial

    
def debug(func=None, prefix=""):
    if func is None:
        return partial(debug, prefix=prefix)
    msg = prefix + func.__qualname__

    @wraps(func)
    def wrapper(*args, **kwargs):
        print(msg)
        return func(*args, **kwargs)
    return wrapper


@debug
def add(x, y):
    return x + y


print(add(3, 4))
# add
# 7


@debug(prefix="***")
def add(x, y):
    return x + y


print(add(3, 4))

# ***add
# 7

```


#### 类装饰器(Class Decorator)

```python

如果类的实例方法都需要同一种装饰器, 用类装饰器


from functools import wraps, partial


def debug(func=None, prefix=""):
    if func is None:
        return partial(debug, prefix=prefix)
    msg = prefix + func.__qualname__

    @wraps(func)
    def wrapper(*args, **kwargs):
        print(msg)
        return func(*args, **kwargs)
    return wrapper


def debugmethods(cls):
    for name, val in vars(cls).items():
        if callable(val):
            setattr(cls, name, debug(val))
    return cls


class T(object):

    @debug
    def a(self):
        print('a')

    @debug
    def b(self):
        print('b')

    @classmethod    # Not wrapped
    def c(cls):
        print('c')
        
    @staticmethod   # Not wrapped
    def d():
        print('d')

    @classmethod    
    @debug
    def e(cls):
        print('c')
        
    @staticmethod   
    @debug
    def f():
        print('d')


t = T()
t.a()
t.b()
t.c()
t.d()
t.e()
t.f()



@debugmethods
class T(object):

    def a(self):
        print('a')

    def b(self):
        print('b')

    @classmethod    # Not wrapped
    def c(cls):
        print('c')
        
    @staticmethod   # Not wrapped
    def d():
        print('d')

        
t = T()
t.a()
t.b()
t.c()
t.d()

```

#### 18 元类(metaclass) 

```python

如果子类的实例方法都需要同一种装饰器, 基类使用元类


from functools import wraps, partial


def debug(func=None, prefix=""):
    if func is None:
        return partial(debug, prefix=prefix)
    msg = prefix + func.__qualname__
        
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(msg)
        return func(*args, **kwargs)
    return wrapper
        
    
def debugmethods(cls):
    for name, val in vars(cls).items():
        if callable(val):
            setattr(cls, name, debug(val))
    return cls


@debugmethods
class Base(object):
    pass
 
 
@debugmethods
class Spam(Base):
    pass

    
@debugmethods
class Grok(Spam):
    pass


@debugmethods
class Mondo(Grok):
    pass


Solution: A Metaclass

class debugmeta(type):
    def __new__(cls, clsname, bases, clsdict):
        clsobj = super().__new__(cls, clsname, bases, clsdict)
        clsobj = debugmethods(clsobj)
        return clsobj
    

class Base(metaclass=debugmeta):
    pass


class Spam(Base):
    pass


class Grok(Spam):
    pass


class Mondo(Grok):

    def t(self):
        print('t')


m = Mondo()
m.t()

```

#### 19 进程(process) 


#### 启动一个子进程

```python

import os
from multiprocessing import Process


def t(name):
    print("Child process {0} {1}".format(os.getpid(), name))


if __name__ == "__main__":
    print("Parent process {0}".format(os.getpid()))
    p = Process(target=t, args=("test",))
    print("Child process will start")
    p.start()  # 启动子进程
    p.join()   # 等待子进程结束后再继续往下运行
    print("Child process end")

```


#### 进程池

```python

import os
import time
import random
from multiprocessing import Pool


def task(name):
    print("Run task {0}, {1}".format(name, os.getpid()))
    start = time.time()
    time.sleep(random.random()*3)
    end = time.time()
    print("Task {0} runs {1} seconds".format(name, (end - start)))


if __name__ == "__main__":
    print("Parent process {0}".format(os.getpid()))
    p = Pool()          # 进程池默认数量和电脑有关, 4核CPU，就是进程池默认是4
    for i in range(5):  # 测试的电脑4核CPU，这里起5个子进程，会有一个进程等待其它进程执行完后执行
        p.apply_async(task, args=(i,))
    print("Waiting for all subprocesses done...")
    p.close()  # close 后就不能继续添加新的Process了
    p.join()   # 等待子进程结束后再继续往下运行
    print("All subprocesses done")


# Parent process 19271
# Waiting for all subprocesses done...
# Run task 0, 19272
# Run task 1, 19273
# Run task 2, 19274
# Run task 3, 19275
# Task 1 runs 1.0092787742614746 seconds
# Run task 4, 19273
# Task 3 runs 1.3692305088043213 seconds
# Task 0 runs 1.4714231491088867 seconds
# Task 2 runs 1.9193198680877686 seconds
# Task 4 runs 1.993635892868042 seconds
# All subprocesses done

```


#### 子进程(subprocess)

```python

import subprocess

p = subprocess.Popen(["echo", "Hi"], stdout=subprocess.PIPE)
out, error = p.communicate()
print(out.decode("utf-8"))

# Hi

```

#### 20 线程(thread) 

#### 单个线程

```python

from time import time


def factorize(num):
    for i in range(1, num+1):
        if num % i == 0:
            yield i


nums = [2139079, 1214759, 1516637, 1852285]
start = time()
for num in nums:
    list(factorize(num))
end = time()
print(end-start)


# 0.4892253875732422

```


#### 多个线程

```python

from time import time
from threading import Thread


class FactorizeThread(Thread):

    def __init__(self, num):
        super().__init__()
        self.num = num 

    def factorize(self, num):
        for i in range(1, num+1):
            if num % i == 0:
                yield i

    def run(self):
        self.factors = list(self.factorize(self.num))


nums = [2139079, 1214759, 1516637, 1852285]
start = time()
threads = []
for num in nums:
    thread = FactorizeThread(num)
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()
end = time()
print(end-start)

# 0.4903602600097656   因为GIl的原因无法真正并行计算

```


#### 单线程处理阻塞式IO

```python

from time import time
from select import select


def slow():
    select([], [], [], 0.1)


start = time()
for _ in range(5):
    slow()
end = time()
print(end-start)

# 0.5008544921875

```


#### 多线程处理阻塞式IO

```python

from time import time
from select import select
from threading import Thread

    
def slow():
    select([], [], [], 0.1)


start = time()
threads = []
for _ in range(5):
    thread = Thread(target=slow)
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()
end = time()
print(end-start)

# 0.10056090354919434   处理阻塞式IO速度快5倍
# Python 还有内置的 asyncio 模块来处理阻塞式IO

```


#### 多线程之间数据竞争

```python

from threading import Thread


num = 0


def buy():
    global num
    for i in range(1000000):
        num += 1
        
        
threads = []
for _ in range(2):
    thread = Thread(target=buy)
    threads.append(thread)
    thread.start()
    
for thread in threads:
    thread.join()
print(num)

# 1351674  期望是 2000000 

```


#### 多线程用Lock避免数据竞争

```python

from threading import Thread
from threading import Lock


num = 0 
lock = Lock()


def buy():
    global num 
    for i in range(1000000):
        with lock:
            num += 1


threads = []
for _ in range(2):
    thread = Thread(target=buy)
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
print(num)


# 2000000

```

#### 21 队列(queue) 


#### 在获取端阻塞

```python

from queue import Queue
from threading import Thread

queue = Queue()


def consumer():
    print("Consumer waiting")
    queue.get()                # 会阻塞等待put()
    print("Consumer done")


thread = Thread(target=consumer)
thread.start()
print("Producer putting")
queue.put(object())
thread.join()
print("Producer Done")


# Consumer waiting
# Producer putting
# Consumer done
# Producer Done

```


#### 在添加端阻塞

```python

from queue import Queue
from threading import Thread
import time

queue = Queue(1)  # 这里把缓冲区设为1, 
                  # 意味着第一个put后，如果没有被消耗
                  # 会阻塞在第二个put那里

def consumer():
    time.sleep(0.1)  # 在get前，率先put, 然后阻塞在第二个put上
    queue.get()
    print("Consumer got 1")
    queue.get()
    print("Consumer got 2")


thread = Thread(target=consumer)
thread.start()
queue.put(object())
print("Producer put 1")
queue.put(object())
print("Producer put 2")
thread.join()
print("Producer done")

# Producer put 1
# Consumer got 1
# Producer put 2
# Consumer got 2
# Producer done

```


#### 追踪工作进度

```python

from queue import Queue
from threading import Thread

queue = Queue()


def consumer():
    print("Consumer waiting")
    queue.get()
    print("Consumer done")
    queue.task_done()


Thread(target=consumer).start()  # 线程不需要调用join方法
queue.put(object())
print("Producer waitting")
queue.join()                     # queue 调用join，等待结束即可
print("Producer done")

# Consumer waiting
# Producer waitting
# Consumer done
# Producer done

```

#### 22 协程(coroutine) 

```python

def consumer():
    result = 9
    while True:
        result = yield result
c = consumer()
print(c.send(None))           # 启动生成器
print(c.send(1))              # 传值给生成器
print(c.send(2))
c.close()                     # 结束生成器

# 9
# 1
# 2

```

#### 23 异步IO(asyncio) 

```python


asyncio是Python 3.4版本引入的标准库，

直接内置了对异步IO的支持。

import asyncio
  
  
@asyncio.coroutine
def hello(n):
    print("Hello world!")
    r = yield from asyncio.sleep(n)
    print("Hello again!")
    

loop = asyncio.get_event_loop()
tasks = [hello(5), hello(2)]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()


从Python 3.5开始引入了新的语法async和await

import asyncio
  
  
async def hello(n):
    print("Hello world!")
    r = await asyncio.sleep(n)
    print("Hello again!")


loop = asyncio.get_event_loop()
tasks = [hello(5), hello(2)]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()

```

### Algorithm 

#### 1 冒泡

```python

时间复杂度 n^2

nums = [5, 3, 10, 23, 12]

for i in range(len(nums)):
    for j in range(i):
        if nums[i] < nums[j]:
            nums[i], nums[j] = nums[j], nums[i]
print(nums)

# [3, 5, 10, 12, 23]

```

#### 2 快排

```python


理想时间复杂度logN,但可能出现最坏情况n^2,

可以通过随机化数据，避免最坏的情况。

空间复杂度 1

nums = [5, 3, 10, 23, 12]


def move(nums, low, high):
    i, j = low, low+1
    while j <= high:
        if nums[j] < nums[low]:
            i += 1 
            nums[i], nums[j] = nums[j], nums[i]
        j += 1 
    nums[low], nums[i] = nums[i], nums[low]
    return i


def quick_sort(nums, low, high):
    if low < high:
        index = move(nums, low, high)
        quick_sort(nums, low, index-1)
        quick_sort(nums, index+1, high)


quick_sort(nums, 0, len(nums)-1)
print(nums)

# [3, 5, 10, 12, 23]

```


#### 3 堆排

```python


时间复杂度 logN

空间复杂度 1

nums = [5, 3, 10, 23, 12]


def flow_up(nums, start, end):
    root = start
    while True:
        child = 2*root + 1
        if child > end: 
            break
        if child+1 <= end and nums[child] < nums[child+1]:
            child = child+1
        if nums[root] < nums[child]:
            nums[root], nums[child] = nums[child], nums[root]
            root = child
        else:
            break


def heap_sort(nums):
    for start in range((len(nums)-2)//2, -1, -1): 
        flow_up(nums, start, len(nums)-1)

    for end in range(len(nums)-1, 0, -1): 
        nums[end], nums[0] = nums[0], nums[end]
        flow_up(nums, 0, end-1)


heap_sort(nums)
print(nums)

```

#### 4 并排

```python


时间复杂度 logN

空间复杂度 n

nums = [5, 3, 10, 23, 12]
            
            
def merge(left, right):
    i, j = 0, 0
    result = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]
    return result


def binary(nums):
    if len(nums) <= 1:
        return nums
    else:
        index = len(nums)//2
        left = binary(nums[:index])
        right = binary(nums[index:])
        return merge(left, right)


print(binary(nums))

```

#### 5 斐波纳挈 

```python

def fib(n):
    a, b = 1, 0
    i = 0
    while i < n:
        print(a)
        a, b = a+b, a
        i += 1


fib(4)

```

#### 6 链表成对调换 

```python

class Node(object):

    def __init__(self, value=None, node=None):
        self.value = value
        self.next = node
        
    
root = Node(1, Node(2, Node(3, Node(4))))

        
def swap(node):
    if node and node.next:
        next_node = node.next
        node.next = swap(next_node.next)
        next_node.next = node
        return next_node 
    return node
    

new_root = swap(root)


def display(node):
    if node:
        print(node.value)
    if node and node.next:
        display(node.next)


display(new_root)

```

#### 7 单链表逆置 

```python

class Node(object):

    def __init__(self, value=None, node=None):
        self.value = value
        self.next = node
    
        
root = Node(1, Node(2, Node(3, Node(4))))
        
        
def reverse(node):
    if node:
        pre = node
        cur = node.next
        pre.next = None
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        return pre
    else:
        return node
    
        
new_root = reverse(root)
        

def display(node):
    if node:
        print(node.value)
    if node and node.next:
        display(node.next)


display(new_root)

```

#### 8 二叉树 

```python

class Node(object):

    def __init__(self, value=None, left=None, right=None):
        self.left = left
        self.right = right
        self.value = value
        
        
root = Node(2, Node(1), Node(5))


def display(node):
    if node and node.left:
        display(node.left)
    if node and node.value:
        print(node.value)
    if node and node.right:
        display(node.right)
        

display(root)

```



### Mysql 


#### MyISAM 

```python

适合于一些需要大量查询的应用，但其对于有大量写操作并不是很好

甚至你只是需要update一个字段，整个表都会被锁起来，

而别的进程，就算是读进程都无法操作直到读操作完成

对于 SELECT COUNT(*) 这类的计算是超快无比的

```

#### InnoDB 

```python

是一个非常复杂的存储引擎，对于一些小的应用，它会比 MyISAM 还慢

他是它支持“行锁” ，于是在写操作比较多的时候，会更优秀

InnoDB要求表必须有主键（MyISAM可以没有）

不建议使用过长的字段作为主键，因为所有辅助索引都引用主索引，

过长的主索引会令辅助索引变得过大

```

#### 常用命令 

```python

SELECT * FROM table_name

SELECT column1,column2 FROM table_name

INSERT INTO table_name VALUES (值1,值2,...)

INSERT INTO table_name ( 列1,列2)  VALUES ( 值1,值2)

UPDATE table_name SET column = new_value WHERE 条件

DELETE FROM table_name WHERE 条件

```



### Linux 


#### 常用命令 

```python

ps aux | grep  uwsgi

可以看出主进程

ps -ef | grep uwsgi

```
