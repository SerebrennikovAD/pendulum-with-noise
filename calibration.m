%Вначале запускаем эту программу, подгрузятся первые 4 строки, если выдаст ошибку то это не важно, просто еще не создались нужные файлы.
%Далее открываем файл rlSimplePendulumModel2.slx и запускаем, подав на
%управление константу. 
%Затем снова запускаем этот файл. 
mean1=0;
mean2=0;
std1 = 0;
std2 = 0;
%первые 4 строки нужны для того, чтобы стало возможным запустить программу
%в симулинке, там она создаст файл  data1.mat. 
%Чтобы симулинк посчитал средние числа на управление можно подать один большой сигнал. Затем нужно будет вернуться
%сюда и запустить этот файл еще раз. Затем вернуть управление от обученного агента
%и запустить в симулинке заново. Там применятся переменные рассчитанные
%здесь.
load data1.mat;

mean1 =sum(ans1(2,:))/numel(ans1(2,:));%среднее значение по первому сигналу
mean2 =sum(ans1(3,:))/numel(ans1(3,:));
std1 = std(ans1(2,:));%среднеквадратическое отклонение по первому сигналу
std2 = std(ans1(3,:));
%Чтобы не переназначать гипермараметры обучения пришлось
%ввести эти коэффициенты, иначе пришлось бы переправлять 
%блоки с вознаграждением. Смысл их в том, что когда мы все данные делаем
% с дисперсией от -1 до 1, со средним значением 0, то награда тоже
% изменяется. И формулы которые там есть перестают работать как нужно. 
k1=-0.786163461742378+0.004318682596400;%эти коэффициенты призваны изменить данные так, чтобы значения дисперсии и среднее значение оказалось таким же как в оригинальной программе
k2=-0.469851625928589-0.003863293718851;
k3=0.216201826289573/0.702189461873458;
k4=0.339579901511123/0.712980660159183;
std1=std1*k3;
std2=std2*k4;
mean1=mean1+k1*std1/0.216201826289573; 
mean2=mean2+k2*std2/0.339579901511123;





save data2.mat mean1 mean2 std1 std2