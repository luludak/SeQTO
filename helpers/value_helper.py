class ValueHelper:

    def __init__(self):
        pass

    def is_float(self, val):
        try:
            float(val)
        except ValueError:
            return False
        else:
            return True

    def is_int(self, val):
        try:
            int(val)
        except ValueError:
            return False
        else:
            return True