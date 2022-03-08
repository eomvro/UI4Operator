from eunjeon import Mecab
from konlpy.tag import Komoran

class Chuncker:
    def __init__(self):
        self.tagger = Komoran()
        #self.tagger = Mecab()
        self.Bi_character_feature = []

    def get_feautre(self, query):
        self.Bi_character_feature = []

        TKs = self.tagger.morphs(query)
        pos = self.tagger.pos(query)

        '''
        komoran으로 한국어 형태소 처리 
        TKs 는 query의 형태소 반환 (안녕하세요 / 만나 / 아서/ 반갑 / 어요)
        pos 는 형태소와 그 무너ㅑ.. 이름 반환 (안녕하세요.nnp / 만나.vv / 아서.ec / 반갑.va / 어요.ec)
        '''

        #query 의 형태소 개수만큼
        for i in range(len(TKs)):

            try:
                #pos에서 형태소 이름에 N이 있으면 0, 없으면 -1 (N들어가면 대부분 NNP, NP, NNG)  == 형태소 이름에 N이 있으면
                if str(pos[i][1]).find('N') != -1:
                    # 만들어놓은 bi_character_feature에 형태소 추가 (이름말고!)
                    self.Bi_character_feature.append(TKs[i])
            except:
                None

    def get_chunk_score(self, paragraph):
        score = 0
        #앞서 만들어놓은(N 들어가는 형태소 추가한) bi_character_feature에 있는 원소들 상대로
        #약간 similarity 느낌,,?
        for ch_feat in self.Bi_character_feature:
            if paragraph.find(ch_feat) != -1:
                score += 1

        if len(self.Bi_character_feature) == 0:
            return 1

        return score / len(self.Bi_character_feature)

