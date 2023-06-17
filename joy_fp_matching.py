import json
import re

joy_db = "/Users/student/Downloads/Joy/joy_db_processed.json"
our_db = "/Users/student/Downloads/Joy/joy_fp_created.json"

with open(joy_db, 'r') as joy:
    joy_read = []
    for line_joy in joy:
        # print("joy line "+str(joyline))
        joy_read.append(json.loads(line_joy))

with open(our_db, 'r') as ours:
    with open("/Users/student/Downloads/Joy/joy_OS_result.json", 'w') as outfile:
        ourline=1
        for line_ours in ours:
            print("ourline " + str(ourline))
            obj = {}
            data_ours = json.loads(line_ours)
            our_subs = re.findall(r'\((.*?)\)', data_ours["joy_fp"])
            # joyline=1
            # with open(joy_db, 'r') as joy:
            for item in joy_read:
                flag = True
                # print("joy line "+str(joyline))
                joy_subs = re.findall(r'\((.*?)\)', str(item["str_repr"]))
                # False if lengths different, move onto next line in Joy
                if (len(our_subs) != len(joy_subs)):
                    flag=False
                    continue

                loop_count = 1
                # for each corresponding () compare values
                for joy, ours in zip(joy_subs, our_subs):
                    # TLS version, exact match. If False move onto next line in Joy
                    if loop_count == 1:
                        if joy != ours:
                            flag=False
                            break
                    
                    # cipher suites, exact match. If False move onto next line in Joy
                    if loop_count == 2:
                        if joy != ours:
                            flag=False
                            break
                    
                    # extensions list
                    if loop_count >= 3:
                        joy_name = joy[:4]
                        our_name = ours[:4]
                        if joy_name != our_name:
                            flag=False
                            break

                        # extensions with no value in our database, match for presence
                        # check 45 (psk_key_exchange_mode - is this client_key_exchange?)
                        # ["65281", "0", "35", "21", "51", "23", "5", "17", "49", "41", "34", "28", "42", "45", "15", "50", "18", "13172", "22", "65284", "27", "17513", "24", "44", ]
                        if joy_name in ['000a', '000b', '000d', '002b', 'ff01', '0000', '0023', '0015', '0033', '0017', '0005', '0011', '0031', '0029', '0022', '001c', '002a', '002d', '000f', '0032', '0012', '3374', '0016', 'ff04', '001b', '4469', '0018', '002c', '0010']:
                            loop_count = loop_count+1
                            continue

                        # 20 (server_certificate_type) doesn't exist in joy therefore discard
                        # extensions that need exact value matching
                        # 10, 11, 13, 43, 16
                        # elif joy_name in ['000a', '000b', '000d', '002b']:
                        #     if joy[4:] != ours[4:]:
                        #         flag=False
                        #         break
                        
                        # what to do with extensions that are unassigned, reserved, etc? currently check if present, but maybe no need? delete from our db?

                        # if joy_name is not one of the extensions in our db
                        
                        else:
                            flag=False
                            break
                    
                    loop_count = loop_count+1
                # found matching Joy fp
                if flag==True:
                    obj["our_fp"] = data_ours["our_fp"]
                    obj["joy_fp"] = data_ours["joy_fp"]
                    obj["joy_process_info"] = item["process_info"]
                    outfile.write(json.dumps(obj) + '\n')
                    break

            ourline=ourline+1    