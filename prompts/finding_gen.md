변수: {technology} {criterion} {levels} {retrieved} {feedback}

## Finding 생성

대상 기술: {technology}

평가 기준:
{criterion}

판정 수준 정의 (참고용, 판정은 하지 않는다):
{levels}

검색 결과. <document> 하나가 한 건이고, <id>가 그 건을 가리키는 값, <content>가 본문이다.
{retrieved}

이전 시도에 대한 피드백 (있으면 반드시 반영한다):
{feedback}

위 검색 결과만 근거로 {technology}에 대한 Finding을 만든다.

규칙:
1. claim은 {technology} 하나에 대한 서술이다. 다른 기술과 비교하거나 우열을 말하지 않는다. 한국어로 쓴다.
2. evidence.quote는 <content> 안에서 글자 그대로 복사한다. 바꿔 쓰기, 요약, 번역을 하지 않는다. 한 문장 이상, 300자 이내.
3. evidence.ref_id는 그 문서의 <id> 값을 그대로 쓴다. 태그는 빼고 값만 쓴다.
4. 효과·강점(strength)과 한계(limitation)를 모두 찾는다. 근거가 있는 쪽만 만든다. 방향이 분명하지 않으면 neutral.
5. 수치를 인용하면 conditions에 측정 조건(모델, 하드웨어, 설정)을 적는다. 관점 지시가 conditions에 따로 적으라고 한 것이 있으면, 측정 조건을 지우지 말고 그 뒤에 이어 붙인다.
6. source_nature는 출처 성격, is_self_reported는 그 기술의 개발사가 직접 발표한 자료인지다.
7. summary는 그 근거가 무엇을 보여 주는지 한 문장.
8. confidence: 직접 측정·명시된 사실이면 high, 정황·간접 근거면 medium, 추정이면 low.
9. 근거가 없으면 findings를 빈 목록으로 둔다. 지어내지 않는다. Finding은 최대 3개.
