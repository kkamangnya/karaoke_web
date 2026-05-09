# karaoke_web

브라우저와 Python 양쪽에서 사용할 수 있는 노래방 음원 피치 시프터 프로젝트입니다.

## 포함 파일

- `browser_karaoke_pitch_shifter.html`: 브라우저 전용 단일 페이지 버전
- `browser_karaoke_pitch_shifter_best.html`: 고음질 알고리즘과 개선된 UI가 적용된 브라우저 버전
- `karaoke_pitch_shifter.py`: Python CLI 및 GUI 버전
- `WORKLOG.md`: 작업 타임라인 기록

## 브라우저 버전

`browser_karaoke_pitch_shifter_best.html`을 열면 다음 기능을 사용할 수 있습니다.

- MP3 / WAV 업로드
- 반음 단위 피치 조절
- 자동 키 분석
- MP3 다운로드
- 드래그 앤 드롭
- 진행률 표시
- 알고리즘 설명 팝오버

## Python 버전

`karaoke_pitch_shifter.py`는 다음 기능을 제공합니다.

- MP3 / WAV 입력
- 반음 단위 피치 변경
- 고정 속도 유지
- 키 자동 분석
- 보컬 완화
- 노이즈 감소
- CLI / GUI 실행

## 실행 예시

```bash
python karaoke_pitch_shifter.py -i input.mp3 -o output.wav -s 2
python karaoke_pitch_shifter.py --gui
```

브라우저 버전은 로컬 파일로 직접 열 수 있습니다.
