
def create_name(row):
    img_name: str = row['URL'].lower()
    new_img_name = img_name.strip().replace('http://smartdots.ices.dk/sampleimages/', '').replace('/', '_')
    return new_img_name


def is_label_ok(row):
    # print(row)
    row_label: str = row['ModalAge_AllReaders']
    age = int(row_label)
    if age < 0:
        return True
    return True

# def create_name(row):
#     img_name: str = row['ImageName'].lower()
#     img_new_name = os.path.splitext(img_name)[0] + '.jpg'
#     return img_new_name


def unify_label(row, num_classes):
    label = num_classes * [0.0]
    row_label: str = row['Stats']
    splits = row_label.replace(' ', '').split(',')
    for split in splits:
        sp = split.split(':')
        class_number = sp[0]
        if int(class_number) >= num_classes:
            return None
        observ_amount = sp[1]
        label[int(class_number)] = int(observ_amount)
    return label


def inverse_weights(lst):
    sum = 0.0
    for l in lst:
        sum = sum + 1/l
    return [1/(l*sum) for l in lst]

