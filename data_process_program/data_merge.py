import os
import csv
import glob


INPUT_PATH = '../data_collection_program'
OUTPUT_PATH = './all_dataset.csv'
SUBDIRS_TO_PROCESS = ['normal', 'falling', 'fake']


# -------- 통합 실행 --------

def main():
    print("CSV 통합 시작")

    # 출력 폴더 생성
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    header = ['index', 'filename'] + [f'v{i}' for i in range(300)] + ['label']

    index_counter = 0

    try:
        with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)

            # 지정된 폴더 순서대로 처리
            for subdir in SUBDIRS_TO_PROCESS:
                target_dir = os.path.join(INPUT_PATH, subdir)
                print(f"폴더 처리: {subdir}")

                file_list = sorted(glob.glob(os.path.join(target_dir, '*.csv')))
                if not file_list:
                    print(f"  {subdir} 폴더에 CSV 없음")
                    continue

                added = 0

                for file_path in file_list:
                    filename = os.path.basename(file_path)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            row = next(csv.reader(infile))

                        new_row = [index_counter, filename] + row
                        writer.writerow(new_row)

                        index_counter += 1
                        added += 1

                    except Exception as e:
                        print(f"오류: {filename} 처리 실패: {e}")

                print(f"{added}개 추가 완료")

        print(f"통합 완료. 총 {index_counter}개 행 저장됨.")

    except Exception as e:
        print(f"통합 실패: {e}")


if __name__ == '__main__':
    main()