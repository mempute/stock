dynagen과 generic 시계열 망을 사용하여 주가 예측한 결과

7일 후를 예측하여 파란색은 예측, 적색은 실주가
학습한 주가는 완벽한 선행을 나타내며
예측 주가는 kg케미칼 lg전자, .. 등에서 보듯이 대략 50일 정도
실 주가와 유사한 패턴을 나타냄
lstm이나 기타 시계열 망을 사용했을 경우 학습된 주가조차 lagging된 그라프를 보여주는
것과 비교하여 확연한 차이

실행 방법
1. stock_data3.zip 압축 해제
2. python stock_multis.py 2 1 0 7 6 3 0.7 실행
3. 학습 수행할려면 
4. python stock_multis.py 1 1 0 7 6 45 0.7 1000 실행
5. stock_data3의 각 주가 종목 아이디를 위 옵션 45에 대채하여 종목 별로 학습

유의사항

tesla v100장비에서 훈련한 모델, pc급에서는 메모리가 부족하여 malloc error 발생될 수 있음.
