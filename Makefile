# Makefile

# 최종 실행 파일 이름
PROJECT_NAME = lane_detection

# 사용할 컴파일러
CXX = g++

# 빌드 옵션
CXXFLAGS = -std=c++17 -Iinclude `pkg-config --cflags opencv4`
LDFLAGS  = `pkg-config --libs opencv4`

# 빌드 대상 소스
SOURCES = $(wildcard src/*.cpp)

# OBJECTS (자동으로 .o 확장자를 대응)
OBJECTS = $(SOURCES:.cpp=.o)

# 기본 빌드 규칙
$(PROJECT_NAME): $(OBJECTS)
	$(CXX) -o $@ $(OBJECTS) $(LDFLAGS)

# 각 .cpp -> .o 규칙
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 청소 규칙
clean:
	rm -f $(PROJECT_NAME) $(OBJECTS)
