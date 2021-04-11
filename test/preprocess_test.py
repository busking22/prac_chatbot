import sys

sys.path.append(".")

from utils.preprocess import Preprocess


sent = """
SWM 연수과정의 많은 부분이 자발적 참여로 이루어지고, 참여의 목적을 달성하고 연수과정을 즐겁게 보내려면, 시너지를 낼 수 있는 팀원, 지속적으로 흥미를 유발하고, 도전 하고 싶은 프로젝트를 선정하는 일이 정말 중요하겠죠. 
"""

p = Preprocess()
pos = p.pos(sent)

ret = p.get_keywords(pos)
print(ret)
ret = p.get_keywords(pos, True)
print(ret)