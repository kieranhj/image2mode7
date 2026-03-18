CFLAGS :=
LFLAGS :=

CC := g++
LINK := g++

BUILD_DIR     := build

all: $(BUILD_DIR)/image2mode7

$(BUILD_DIR)/%.o: image2mode7/%.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $< -c -o $@

$(BUILD_DIR)/image2mode7: $(BUILD_DIR)/image2mode7.o
	$(LINK) $(LFLAGS) $< -o $@
