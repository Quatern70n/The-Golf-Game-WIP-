import shelve
from pprint import pprint


class Saver:  # Класс для работы с сохранениями
    def __init__(self, filename):
        self.filename = filename

    def save(self, name, values):
        with shelve.open(self.filename) as file:
            d = {}
            try:
                d = file[name]
            except KeyError:
                self.write(name, values)
            for key, val in values.items():
                d[key] = val
            file[name] = d

    def write(self, name, value):
        with shelve.open(self.filename) as file:
            file[name] = value

    def read(self, name, param=None):
        with shelve.open(self.filename) as file:
            try:
                if param:
                    return file[name][param]
                return file[name]
            except KeyError:
                return None

    def filter_save(self, name, values, func=lambda x, y: x > y):
        with shelve.open(self.filename) as file:
            d = {}
            try:
                d = file[name]
            except KeyError:
                self.write(name, values)
            for key, val in values.items():
                if func(val, d[key]):
                    d[key] = val
            file[name] = d


class TextSaver:  # Класс для работы с текстовыми сохранениями (моделей)
    def __init__(self, filename, delimiter='.\n', val_delimiter='\nEND\n'):
        self.filename = filename
        self.delimiter = delimiter
        self. val_delimiter = val_delimiter
        self.file = {}
        self.update()

    def update(self):  # Если файл был изменен то нужно перезагрузить информацию этой функцией
        new_file = {}
        with open(self.filename, 'r') as f:
            reader = f.read()
            lst = reader.split(self.delimiter)
            for i in lst:
                pair = i.split('\n')
                val = '\n'.join(pair[1:])
                new_file[pair[0]] = self.splitter(val)
        self.file = new_file

    def splitter(self, text):
        data = text.split(self.val_delimiter)
        return self.vector_split(data[0]), self.build_split(data[1])

    def vector_split(self, text):
        vectors = []
        for i in text.split('\n'):
            data = i.replace('[', '').replace(']', '').split(', ')
            if data != ['']:
                format_data = [data[0], data[1], data[2], [data[3], data[4], data[5]],
                               [data[6], data[7], data[8]]]
                vectors.append(format_data)
        return vectors

    def build_split(self, text):
        text = text[1:-1].replace('\n', ' ')
        build = [list(map(int, i.split(', '))) for i in text.strip('[]').split('], [')]
        return build

    def get_file(self):  # Возвращает словарь всего файла
        return self.file

    def get_obj(self, name):  # Возвращает данные по названию из файла
        return self.file[name]

    def save(self, filename, name):  # сохраняет информацию в базу данных
        saver = Saver(filename)
        saver.write(name, self.file[name])
