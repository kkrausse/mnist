#derived from this link:
#https://sites.google.com/site/michaelsafyan/software-engineering/how-to-write-a-makefile

program_NAME := nets
program_SRCS := $(wildcard *.c)
program_OBJS := ${program_SRCS:.c=.o}

#defines the default c compiler
CC := gcc

#defines flags to be passed to the c compiler
CFLAGS := -Wall -Werror

all: $(program_NAME)

$(program_NAME): $(program_OBJS)
	gcc $(program_OBJS) -o $(program_NAME)

clean:
	$(RM) $(program_NAME)
	$(RM) $(program_OBJS)
