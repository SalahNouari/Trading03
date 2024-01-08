
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



if __name__ == '__main__':
    print("===== START =====")

    print(torch.manual_seed(1))

    lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
    inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

    # initialize the hidden state.
    hidden = (torch.randn(1, 1, 3),
              torch.randn(1, 1, 3))
    for i in inputs:
        # Проходим по последовательности по одному элементу за раз.
        # после каждого шага hidden содержит скрытое состояние.
        out, hidden = lstm(i.view(1, 1, -1), hidden)

    # в качестве альтернативы мы можем выполнить всю последовательность сразу.
    # первое значение, возвращаемое LSTM, — это все скрытые состояния
    # последовательность. второе - это просто самое последнее скрытое состояние
    # (сравните последний фрагмент "out" и "hidden" ниже, они одинаковые)
    # Причина этого в том, что:
    # "out" даст вам доступ ко всем скрытым состояниям в последовательности
    # "hidden" позволит вам продолжить последовательность и выполнить обратное распространение,
    # передав его в качестве аргумента в lstm позже
    # Добавляем дополнительное 2-е измерение

    inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
    out, hidden = lstm(inputs, hidden)
    print(out)
    print(hidden)


    k=1
