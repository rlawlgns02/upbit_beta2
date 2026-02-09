"""
오래된 LSTM 모델 체크포인트 정리 스크립트
학습 완료 후 중간 에폭 파일들을 정리하여 디스크 공간 확보
"""
import os
import glob
from pathlib import Path


def cleanup_old_checkpoints(models_dir: str = 'models', keep_best_only: bool = False):
    """오래된 체크포인트 파일 정리

    Args:
        models_dir: 모델 디렉토리 경로
        keep_best_only: True이면 best 모델만 유지, False이면 최신 모델도 유지
    """
    if not os.path.exists(models_dir):
        print(f"❌ 모델 디렉토리가 없습니다: {models_dir}")
        return

    print("🧹 LSTM 모델 체크포인트 정리 시작")
    print("=" * 60)

    # 모든 .pt 파일 찾기
    all_files = glob.glob(os.path.join(models_dir, "*.pt"))

    if not all_files:
        print("📁 정리할 모델 파일이 없습니다.")
        return

    # 파일 분류
    to_keep = []  # 유지할 파일
    to_delete = []  # 삭제할 파일

    for file_path in all_files:
        filename = os.path.basename(file_path)

        # 베스트 모델은 항상 유지
        if '_best.pt' in filename:
            to_keep.append(file_path)
        # 최신 모델 유지 여부
        elif filename.endswith('.pt') and '_epoch' not in filename:
            if keep_best_only:
                to_delete.append(file_path)
            else:
                to_keep.append(file_path)
        # 에폭 체크포인트는 삭제 대상
        elif '_epoch' in filename:
            to_delete.append(file_path)
        else:
            to_keep.append(file_path)

    # 삭제할 파일 목록 출력
    if to_delete:
        print(f"\n🗑️  삭제할 파일 ({len(to_delete)}개):")
        total_size = 0
        for file_path in to_delete:
            size = os.path.getsize(file_path)
            total_size += size
            print(f"  - {os.path.basename(file_path)} ({size / 1024:.1f} KB)")

        print(f"\n💾 절약 가능 용량: {total_size / (1024*1024):.2f} MB")

        # 사용자 확인
        confirm = input("\n❓ 정말로 삭제하시겠습니까? (y/N): ").strip().lower()

        if confirm == 'y':
            deleted_count = 0
            for file_path in to_delete:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ 삭제 실패: {os.path.basename(file_path)} - {e}")

            print(f"\n✅ {deleted_count}개 파일 삭제 완료!")
        else:
            print("\n🚫 삭제 취소됨")
    else:
        print("✨ 정리할 체크포인트 파일이 없습니다.")

    # 유지되는 파일 목록
    if to_keep:
        print(f"\n📦 유지되는 파일 ({len(to_keep)}개):")
        for file_path in to_keep:
            size = os.path.getsize(file_path)
            print(f"  - {os.path.basename(file_path)} ({size / 1024:.1f} KB)")

    print("\n" + "=" * 60)
    print("✅ 정리 완료!")


def get_model_info(models_dir: str = 'models'):
    """모델 파일 정보 출력"""
    if not os.path.exists(models_dir):
        print(f"❌ 모델 디렉토리가 없습니다: {models_dir}")
        return

    all_files = glob.glob(os.path.join(models_dir, "*.pt"))

    if not all_files:
        print("📁 모델 파일이 없습니다.")
        return

    print("\n📊 현재 모델 파일 현황")
    print("=" * 60)

    # 코인별로 그룹화
    coin_models = {}
    for file_path in all_files:
        filename = os.path.basename(file_path)

        # 코인명 추출
        if filename.startswith('lstm_'):
            coin = filename.split('_')[1].split('.')[0]
            if '_epoch' in filename:
                coin = filename.split('_')[1]

            if coin not in coin_models:
                coin_models[coin] = {'main': None, 'best': None, 'epochs': []}

            if '_best.pt' in filename:
                coin_models[coin]['best'] = file_path
            elif '_epoch' in filename:
                coin_models[coin]['epochs'].append(file_path)
            else:
                coin_models[coin]['main'] = file_path

    # 출력
    total_size = 0
    for coin, models in sorted(coin_models.items()):
        print(f"\n🪙 {coin.upper()}")

        if models['main']:
            size = os.path.getsize(models['main'])
            total_size += size
            print(f"  ├─ 최신 모델: {os.path.basename(models['main'])} ({size/1024:.1f} KB)")

        if models['best']:
            size = os.path.getsize(models['best'])
            total_size += size
            print(f"  ├─ 베스트 모델: {os.path.basename(models['best'])} ({size/1024:.1f} KB)")

        if models['epochs']:
            epoch_size = sum(os.path.getsize(f) for f in models['epochs'])
            total_size += epoch_size
            print(f"  └─ 체크포인트: {len(models['epochs'])}개 파일 ({epoch_size/1024:.1f} KB)")

    print("\n" + "=" * 60)
    print(f"📊 총 용량: {total_size / (1024*1024):.2f} MB")
    print(f"📁 총 파일 수: {len(all_files)}개")


if __name__ == '__main__':
    import sys
    # Windows 콘솔 인코딩 설정
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("LSTM 모델 체크포인트 관리 도구")
    print()

    # 현재 상태 확인
    get_model_info()

    print("\n\n옵션을 선택하세요:")
    print("1. 에폭 체크포인트만 삭제 (최신 + 베스트 유지)")
    print("2. 베스트 모델만 유지 (최신 모델도 삭제)")
    print("3. 취소")

    choice = input("\n선택 (1-3): ").strip()

    if choice == '1':
        cleanup_old_checkpoints(keep_best_only=False)
    elif choice == '2':
        cleanup_old_checkpoints(keep_best_only=True)
    else:
        print("🚫 취소됨")
