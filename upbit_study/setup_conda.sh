#!/bin/bash
# 업비트 AI 트레이딩 시스템 - Conda 환경 자동 설정 스크립트 (macOS/Linux)

echo "================================================"
echo "업비트 AI 트레이딩 시스템 설치 시작"
echo "================================================"
echo ""

# Conda 설치 확인
if ! command -v conda &> /dev/null
then
    echo "[ERROR] Conda가 설치되어 있지 않습니다!"
    echo ""
    echo "Miniconda 설치:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    exit 1
fi

echo "[1/5] Conda 버전 확인..."
conda --version
echo ""

echo "[2/5] 가상환경 'upbit' 생성 중..."
conda create -n upbit python=3.10 -y
if [ $? -ne 0 ]; then
    echo "[ERROR] 가상환경 생성 실패"
    exit 1
fi
echo "가상환경 생성 완료!"
echo ""

echo "[3/5] 가상환경 활성화..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate upbit
echo ""

echo "[4/5] pip 업그레이드..."
python -m pip install --upgrade pip
echo ""

echo "[5/5] 패키지 설치 중... (몇 분 소요될 수 있습니다)"
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] 패키지 설치 실패"
    exit 1
fi
echo ""

echo "================================================"
echo "설치 완료!"
echo "================================================"
echo ""
echo "다음 명령어로 시작하세요:"
echo "  conda activate upbit"
echo "  python recommend.py"
echo ""
echo "상세 가이드: INSTALL_CONDA.md 참고"
echo "================================================"
