'''
Реализация основных класс-функций оценки достижения критерия остановки.
'''

from .core import StopCaseBase


class LessEqualStopCase(StopCaseBase):
    '''
    Критерий остановки по условию "менее либо равно, чем"
    '''
    def __init__(self, value):
        self.value = value

    def __call__(self, value, **kwargs):
       return value <= self.value


class MoreEqualStopCase(StopCaseBase):
    '''
    Критерий остановки по условию "больше либо равно, чем"
    '''
    def  __init__(self, value):
        self.value = value

    def __call__(self, value, **kwargs):
        return value >= self.value

class NSameStopCase(StopCaseBase):
    '''
    Критерий остановки по условию, что на протяжении нескольких итераций
    значения равны, т.е. не изменяются
    '''
    def __init__(self, n=5):
        self.history = []
        self.n = n

    def __call__(self, value, **kwargs):
        self.history.append(value)
        last = self.history[-self.n:]
        if [value]*self.n == last:
            self.history = []
            return True
        else:
            return False

class ItersStopCase(StopCaseBase):
    '''
    Критерий остановки по условию, что выполнено заданное количество итераций (эпох, шагов и т.д.)
    '''
    def __init__(self, max_iter_count=2):
        self.max_iter_count = max_iter_count

    def __call__(self, **kwargs):
        iters_count = kwargs['iteration']
        
        return iters_count>=self.max_iter_count


