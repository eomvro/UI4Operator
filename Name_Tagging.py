import numpy as np


class Name_tagger:
    def __init__(self):
        countries = open('LCP_COUNTRY.txt', 'r').read().split()
        names = open('PS_NAME (2).txt', 'r').read().split()

        self.country_dic = np.array(countries, dtype='<U20')
        self.name_dic = np.array(names, dtype='<U20')

        self.country_dic.sort()
        self.name_dic.sort()

        print('dic shape', self.country_dic.shape, self.name_dic.shape, self.name_dic.dtype)
        #input()

    def get_name_tag(self, word):
        if self.search(self.country_dic, word) is True:
            return 1

        if self.search(self.name_dic, word) is True:
            return 2

        if self.is_num(word) is True:
            if self.is_time(word) is True:
                return 3
            else:
                return 4

        return 0

    def search(self, dic, word):
        idx = dic.searchsorted(word)
        if dic.shape[0] != idx:
            if word == dic[idx]:
                return True
        return False

    def is_num(self, word):
        if len(word) == 0:
            return False

        num = 0

        c_s = 48
        c_e = 57

        for i in range(len(word)):
            if 48 <= ord(word[i]) <= 57:
                num += 1

        if num / len(word) >= 0.5:
            return True

        return False

    def is_time(self, word):
        time_conj = ['분', '초', '년', '월', '일']

        for conj in time_conj:
            if word.find(conj) != -1:
                return True
        return False