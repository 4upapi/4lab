#30.	Формируется матрица F следующим образом: скопировать в нее А и  если в В количество простых чисел в нечетных столбцах,
# чем произведение чисел по периметру С, то поменять местами  С и В симметрично, иначе С и В поменять местами несимметрично.
# При этом матрица А не меняется. После чего если определитель матрицы А больше суммы диагональных элементов матрицы F,
# то вычисляется выражение: A*AT – K * F, иначе вычисляется выражение (A-1 +G-F-1)*K,
# где G-нижняя треугольная матрица, полученная из А.
# Выводятся по мере формирования А, F и все матричные операции последовательно.

import numpy as np
import matplotlib.pyplot as plt
print("Введите число N не меньше 6:", end='')
n = int(input())
print("Введите число K:", end='')
k = int(input())
K = k#K в массив для умножения потом
A = np.random.randint(-10,10, size=(n,n),dtype="int64")#создание основной матрицы
F = np.copy(A)#копирование основ. матрицы
kolich = 0#количество чисел, больших К в четных столбцах
suma_el = 0#сумма элементов диагоналей
proizv = 1#произведение чисел
print('Основная матрица')
print(F)
a = [2,3,5,7]
for i in range(0,n//2):
    for j in range(0, n // 2):
        if j%2 == 0 and F[i,j] in a:
            kolich += 1
for i in range(n // 2, n):
    proizv *= F[0, i]
for i in range(1, n // 2):
    proizv *= F[i, n // 2]
for i in range((n // 2) + 1, n):
    proizv *= F[(n // 2) - 1, i]
for i in range(1, (n // 2) - 1):
    proizv *= F[i, n - 1]
if kolich > proizv:
    for i in range(0,n//2):
        for j in range(0,n//2):
            F[i,j],F[i, n-1 - j] = F[i, n-1 - j],F[i,j]
elif kolich <= proizv:
    for i in range(0,n//2):
        for j in range(0,n//2):
            F[i, j], F[i, j+ n//2] = F[i, j+ n//2], F[i, j]
print(F)
opredlitel = int(np.linalg.det(A))#функция для поиска определителя
for i in range(0,n):#сумма диагональных элементов
    suma_el = suma_el + F[i,i]
print(' ')
for i in range(0,n):
    suma_el = suma_el + F[i, n - 1 - i]
print('Определитель A:', opredlitel)
print('Сумма диагональных чисел F:', suma_el)
print('')
if opredlitel > suma_el:
    At = np.transpose(A)#функция транспортирования
    Proizv = A * At
    print("A*A транспонированная")
    print(Proizv)
    print('')
    print('')
    print(" A*A транспонированная - K *F")
    print(Proizv - (K * F))
elif opredlitel < suma_el:
    print("A транспонированная")
    A1 = np.linalg.inv(A)#функция транспортирования
    print(A1)
    print('')
    print('G')
    G = np.copy(np.tril(A))#linalg.inv = обратная матрица
    print(G)
    print('')
    Ft = np.transpose(F)
    print("F -1")
    F1 = np.linalg.inv(F)
    print(F1)
    print('')
    print("(At + G1 - F1)*K")
    ND = (A1 + G - F1)*K
    print(ND)
s1 = [np.mean(abs(F[i, ::])) for i in range(n)]
s1 = int(sum(s1))
fixg, s2 = plt.subplots(2, 2, figsize=(11, 8))
x = list(range(1, n + 1))
for j in range(n):
    y = list(F[j, ::])
    s2[0, 0].plot(x, y, ',-', label=f"{j + 1} строка.")
    s2[0, 0].set(title="График с использованием функции plot:", xlabel='Номер элемента в строке',
                 ylabel='Значение элемента')
    s2[0, 0].grid()
    s2[0, 1].bar(x, y, 0.4, label=f"{j + 1} строка.")
    s2[0, 1].set(title="График с использованием функции bar:", xlabel='Номер элемента в строке',
                 ylabel='Значение элемента')
    if n <= 10:
        s2[0, 1].legend(loc='lower right')
        s2[0, 1].legend(loc='lower right')
mod = [0] * (n - 1)
mod.append(0.1)
sizes = [round(np.mean(abs(F[i, ::])) * 100 / s1, 1) for i in range(n)]
s2[1, 0].set_title("График с ипользованием функции pie:")
s2[1, 0].pie(sizes, labels=list(range(1, n + 1)), explode=mod, autopct='%1.1f%%', shadow=True)
def map(data, row_labels, col_labels, grap3, bar_gh={}, **kwargs):
    da = grap3.imshow(data, **kwargs)
    bar = grap3.figure.colorbar(da, ax=grap3, **bar_gh)
    grap3.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    grap3.set_yticks(np.arange(data.shape[0]), labels=row_labels)
    return da, bar
def annoheat(da, data=None, textcolors=("black", "white"), threshold=0):
    if not isinstance(data, (list, np.ndarray)):
        data = da.get_array()
    gh = dict(horizontalalignment="center", verticalalignment="center")
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            gh.update(color=textcolors[int(data[i, j] > threshold)])
            text = da.axes.text(j, i, data[i, j], **gh)
            texts.append(text)
    return texts
da, bar = map(F, list(range(n)), list(range(n)), grap3=s2[1, 1], cmap="magma_r")
texts = annoheat(da)
s2[1, 1].set(title="Создание аннотированных тепловых карт:", xlabel="Номер столбца", ylabel="Номер строки")
plt.suptitle("Использование библиотеки matplotlib")
plt.tight_layout()
plt.show()

