변수: {criterion_name} {criterion_question} {retrieved}

## 검색 결과 관련성 판정

평가 기준: {criterion_name}
평가 질문: {criterion_question}

검색 결과. <document> 하나가 한 건이고, <id>가 그 건을 가리키는 값, <content>가 본문이다.
{retrieved}

각 결과가 위 평가 질문에 답하는 근거로 쓸 수 있는지 판정한다.
- 관련 있는 결과의 <id> 값만 relevant_ids에 넣는다. 값을 바꾸거나 줄이지 않는다.
- 기준과 대상 기술 양쪽에 관련이 있어야 한다. 일반적인 배경 설명만 있는 결과는 제외한다.
- 목차, 참고문헌 목록, 표 제목만 있는 결과는 제외한다.
- 근거로 쓸 만한 결과가 하나도 없으면 빈 목록을 주고 reason에 무엇이 부족한지 한 문장으로 쓴다.
