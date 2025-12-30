import os 
import json
import getpass
def load_key(keyname):
    script_dir=os.path.dirname(os.path.abspath(__file__))
    file_name=os.path.join(script_dir,'keys.json')
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            keys = json.load(file)
        if keyname in keys:
            return keys[keyname]
        else:
            keyval=getpass.getpass(f'Enter the key value for {keyname}: ').strip()
            keys[keyname]=keyval
            with open(file_name,'w') as file:
                json.dump(keys,file,indent=4)
            return keyval
    else:
        keyval=getpass.getpass(f'Enter the key value for {keyname}: ').strip()
        keys={
            keyname:keyval
        }
        with open(file_name,'w') as file:
            json.dump(keys,file,indent=4)
        return keyval


if __name__=='__main__':
    print(load_key('openai_api_key'))