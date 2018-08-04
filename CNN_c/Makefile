TARGET=nn
SRCS = library.cpp

OBJS = $(SRCS:.c=.o)

$(TARGET):
	g++ -std=c++11 -shared -O3 -DNDEBUG -fPIC -o nn_c.so library.cpp -Wl,-Bstatic -lopenblas -Wl,-Bdynamic -lpthread -Wl,--as-needed
#	g++ -std=c++14 -shared -O2 -fPIC -o nn_c.so library.cpp libopenblas.a -lpthread
	mv -f ./nn_c.so ../
