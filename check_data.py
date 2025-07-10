#!/usr/bin/env python3
# check_data.py

import os
import numpy as np
from pathlib import Path

def analyze_npy_files(data_dir: str = "data/morigirl_50"):
    """npy 파일들을 분석하여 정보를 출력"""
    
    # 데이터 폴더 확인
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ 데이터 폴더가 없습니다: {data_dir}")
        return
    
    # npy 파일 찾기
    npy_files = list(data_path.glob("*.npy"))
    if not npy_files:
        print(f"❌ {data_dir} 폴더에 npy 파일이 없습니다.")
        return
    
    print(f"📁 분석 대상 폴더: {data_dir}")
    print(f"📄 발견된 npy 파일: {len(npy_files)}개")
    print("=" * 80)
    
    total_records = 0
    
    for npy_file in sorted(npy_files):
        print(f"\n🔍 파일 분석: {npy_file.name}")
        print("-" * 50)
        
        try:
            # npy 파일 로드
            data = np.load(npy_file, allow_pickle=True)
            
            print(f"📊 전체 레코드 수: {len(data):,}개")
            total_records += len(data)
            
            if len(data) > 0:
                # 첫 번째 레코드로 컬럼 정보 확인
                first_record = data[0]
                if isinstance(first_record, dict):
                    print(f"📋 컬럼 정보:")
                    for key, value in first_record.items():
                        if key == 'vector':
                            vector_len = len(value) if isinstance(value, (list, np.ndarray)) else 'Unknown'
                            print(f"  - {key}: 벡터 (차원: {vector_len})")
                        else:
                            print(f"  - {key}: {type(value).__name__} (예시: {value})")
                
                # 예시 10개 출력
                print(f"\n📝 예시 데이터 (최대 10개):")
                sample_size = min(10, len(data))
                
                for i in range(sample_size):
                    record = data[i]
                    if isinstance(record, dict):
                        print(f"\n  [{i+1}] 상품 ID: {record.get('product_id', 'N/A')}")
                        print(f"      가격: {record.get('price', 'N/A'):,}원" if record.get('price') else "      가격: N/A")
                        print(f"      모리걸 여부: {'✅ 모리걸' if record.get('is_morigirl', 0) == 1 else '❌ 비모리걸'}")
                        print(f"      판매점수: {record.get('sales_score', 'N/A'):.3f}" if record.get('sales_score') is not None else "      판매점수: N/A")
                        print(f"      1차 카테고리: {record.get('first_category', 'N/A')}")
                        print(f"      2차 카테고리: {record.get('second_category', 'N/A')}")
                        
                        vector = record.get('vector', [])
                        if isinstance(vector, (list, np.ndarray)) and len(vector) > 0:
                            vector_preview = vector[:5] if len(vector) >= 5 else vector
                            print(f"      벡터 (처음 5개): {[f'{v:.3f}' for v in vector_preview]}...")
                        else:
                            print(f"      벡터: 없음")
                
                # 기본 통계 정보
                print(f"\n📈 기본 통계:")
                
                # 모리걸 비율
                morigirl_count = sum(1 for record in data if record.get('is_morigirl', 0) == 1)
                non_morigirl_count = len(data) - morigirl_count
                print(f"  - 모리걸: {morigirl_count:,}개 ({morigirl_count/len(data)*100:.1f}%)")
                print(f"  - 비모리걸: {non_morigirl_count:,}개 ({non_morigirl_count/len(data)*100:.1f}%)")
                
                # 가격 통계
                prices = [record.get('price', 0) for record in data if record.get('price') is not None]
                if prices:
                    print(f"  - 평균 가격: {np.mean(prices):,.0f}원")
                    print(f"  - 최소 가격: {min(prices):,}원")
                    print(f"  - 최대 가격: {max(prices):,}원")
                
                # 판매점수 통계
                scores = [record.get('sales_score', 0) for record in data if record.get('sales_score') is not None]
                if scores:
                    print(f"  - 평균 판매점수: {np.mean(scores):.3f}")
                    print(f"  - 최소 판매점수: {min(scores):.3f}")
                    print(f"  - 최대 판매점수: {max(scores):.3f}")
            
        except Exception as e:
            print(f"❌ 파일 로드 실패: {e}")
    
    print("\n" + "=" * 80)
    print(f"🎉 전체 요약:")
    print(f"  - 총 파일 수: {len(npy_files)}개")
    print(f"  - 총 레코드 수: {total_records:,}개")

if __name__ == "__main__":
    print("🔬 학습용 데이터 분석 시작")
    analyze_npy_files() 