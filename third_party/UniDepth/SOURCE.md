# UniDepth

`unidepth/` 폴더는 [lpiccinelli-eth/UniDepth](https://github.com/lpiccinelli-eth/UniDepth)의 `unidepth` 패키지를 고치지 않고 그대로 옮긴 것입니다.

- 커밋: `8d8cfe4c7ee15297099983607febf0d4f32eb3d6` (2025-05-18)
- 라이선스: CC BY-NC 4.0 (비상업적 이용). 전문은 이 폴더의 `LICENSE`에 있습니다.
- 저자: Luigi Piccinelli 외. 논문 "UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler".

이 저장소에서는 `scripts/open_frontend/depth_models.py`가 UniDepth V2 ViT-S를 불러 씁니다. 가중치(`lpiccinelli/unidepth-v2-vits14`)는 처음 실행할 때 Hugging Face에서 자동으로 받습니다.
