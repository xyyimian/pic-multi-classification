import io, bson

PIC_PATH = './input/train.bson'

def extract():
	data = bson.decode_file_iter(open(PIC_PATH, 'rb'))
	with open('id.txt','w') as f:
		i = 0
		for d in data:
			f.write(str(d['category_id']))
			f.write('\n')
			i += 1
			print(i)
			


if __name__ == "__main__":
    extract()
