def load_data(path_en, path_ru):
    data_ru = open(path_ru, 'r').read().split('\n')
    data_en = open(path_en, 'r').read().split('\n')
    data_en = list(filter(lambda x: len(x) > 0, data_en))
    data_ru = list(filter(lambda x: len(x) > 0, data_ru))

    return data_ru[:100], data_en[:100]