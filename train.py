

if __name__ == '__main__':

    # read it in to inspect it
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    print("length of dataset in characters: ", len(text))

    # let's look at the first 1000 characters
    print(text[:1000])

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(''.join(chars))
    print(vocab_size)


