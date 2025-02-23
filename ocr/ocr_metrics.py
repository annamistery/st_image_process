import re


class OcrMetrics:

    def __init__(self):
        self._data = dict()

    def _wer_score(self, ocr_text, ref_text, to_dict=False):
        # Разбиваем строки на слова
        ref_words = ref_text.split()
        ocr_words = ocr_text.split()

        # Создаем матрицу
        wer_matrix = [[0] * (len(ocr_words) + 1) for _ in range(len(ref_words) + 1)]

        # Инициализируем первую строку и первый столбец
        for i in range(len(ref_words) + 1):
            wer_matrix[i][0] = i
        for j in range(len(ocr_words) + 1):
            wer_matrix[0][j] = j

        # Заполняем матрицу
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(ocr_words) + 1):
                if ref_words[i - 1] == ocr_words[j - 1]:
                    wer_matrix[i][j] = wer_matrix[i - 1][j - 1]  # слова совпадают
                else:
                    wer_matrix[i][j] = min(wer_matrix[i - 1][j] + 1,  # удаление
                                           wer_matrix[i][j - 1] + 1,  # вставка
                                           wer_matrix[i - 1][j - 1] + 1)  # замена

        # Количество ошибок
        substitutions = deletions = insertions = 0
        i, j = len(ref_words), len(ocr_words)

        # Обратный проход для подсчета ошибок
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_words[i - 1] == ocr_words[j - 1]:
                i -= 1
                j -= 1
            elif i > 0 and wer_matrix[i][j] == wer_matrix[i - 1][j] + 1:  # удаление
                deletions += 1
                i -= 1
            elif j > 0 and wer_matrix[i][j] == wer_matrix[i][j - 1] + 1:  # вставка
                insertions += 1
                j -= 1
            else:  # замена
                substitutions += 1
                i -= 1
                j -= 1

        # Общее количество слов в эталонном тексте
        words_total = len(ref_words)

        # Вычисление WER. Избегаем деления на ноль
        wer_score = (substitutions + deletions + insertions) / words_total if words_total > 0 else None
        self._data = {
                'substitutions': substitutions,
                'deletions': deletions,
                'insertions': insertions,
                'words_total': words_total,
                'wer_score': wer_score
            }
        if to_dict:
            return self._data
        return wer_score

    # Перевод в строчные буквы и фильтрация
    @staticmethod
    def _wer_preprocessing(text_str, mode='words'):
        # words - считаем слова
        # digits - считаем цифры
        ref_s = text_str.lower()
        ref_s = ref_s.replace('\n', ' ')  # меняем перенос строки на пробел

        if mode == 'words':
            ref_s = ''.join(filter(lambda x: str.isalpha(x) or x == ' ', ref_s))
        if mode == 'digits':
            ref_s = re.sub(r'[,;:]', '.', ref_s)  # меняем запятые и двоеточия на точки
            # оставляем цифры либо десятичные дроби формата D.D либо даты формата D.D.D
            ref_s = ' '.join(re.findall(r'\b\d+(?:[.]\d+)?(?:\.\d+)?\b', ref_s))
            # print(f'{ref_s=}')
        # ref_s = ref_s.lstrip().rstrip()

        return ref_s

    def calculate(self, input_str, reference_str, mode='words', to_dict=False):
        input_ready = self._wer_preprocessing(input_str, mode=mode)
        reference_ready = self._wer_preprocessing(reference_str, mode=mode)

        return self._wer_score(input_ready, reference_ready, to_dict=to_dict)
