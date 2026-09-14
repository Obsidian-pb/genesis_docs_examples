'''
Утилиты разработки
'''

import time

import requests


def timing(f, msg='', acc=2, return_time=False):
    '''
    Обертка счетчика затраченного времени
    '''
    def _timing(**kwargs):
        # Фиксация стартового времени функции
        st = time.time()

        # Собственно оборачиваемая функция
        r = f(**kwargs)

        # Печать времени работы функции
        ft = time.time()
        td = round(ft-st,acc)
        if msg == '':
            print(f'ВРЕМЯ: {td} сек')
        else:
            print(f'{msg} {td} сек')
        if return_time:
            return *r, td
        return r
    
    return _timing


class Progressbar(object):
    '''
    Прогресс бар для использования в итеративных функциях
    '''
    def __init__(self, maxval, minval=0, bins=40, ok_char='#', est_char='_'):
        self.maxval = maxval
        self.minval = minval
        self.bins = bins
        self.scope = maxval-minval
        self.val=minval
        self.ok_char = ok_char
        self.est_char = est_char

    def __call__(self, **kwargs):
        self.val+=1
        bins_ok = int(self.bins*(self.val/self.scope))
        bins_still = self.bins-bins_ok
        s = "|"+self.ok_char*bins_ok + self.est_char*bins_still + "| " + f'{round(100*self.val/self.scope, 1)}%'
        print(s, end='\r')
        if bins_ok==self.bins:
            print('')


class TimeCheckPoint(object):
    '''
    Таймер

    `acc`:int
        Точность округления
    '''
    def __init__(self, acc:int=2):
        self.acc = acc
        self.prev = time.time()
    
    def __call__(self, switch=True):
        now = time.time()
        diff = round(now - self.prev, self.acc)
        if switch:
            self.prev = now
        return diff
    


class TGMessenger:
    '''
    Класс отправки уведомлений в телеграм-чат

    `token`:str
        Токен бота. Получается у BotFather
    `chat_id`:str
        ID чата в котором происходит общение.
        Для получения id зайти в чат с ботом, написать сообщение,
        после чего вызвать метод check_chat(). Получится id чата.
        В случае ошибки, будет возвращен текст ответа от сервера.
    '''
    def __init__(self, token, chat_id=None):
        self.token = token
        if chat_id is None:
            chat_id = self.check_chat()
        self.chat_id = chat_id

    def send_message(self, message):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage?chat_id={self.chat_id}&text={message}"
        requests.get(url, timeout=5)
        return self
    
    def check_chat(self):
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        jsn = requests.get(url, timeout=5).json()
        try:
            return jsn['result'][0]['message']['from']['id']
        except IndexError:
            print('Разобрать ответ не удалось. Тело ответа:', jsn)
            print('Рекомендуется предварительно написать боту сообщение.')
            return None