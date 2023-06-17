import json

# first two parentheses (TLS version)(cipher suites)
def first_two(fp):
    version = ""
    if fp["version"] == "771":
        version ="(0303)"
    elif fp["version"] == "769":
        version ="(0301)"
    elif fp["version"] == "770":
        version ="(0302)"
    elif fp["version"] == "772":
        version ="(0304)"

    cs = ""
    hex_val = ""
    for cipher_suite in fp["handshake"]["client_hello"]["cipher_suites"]:
        hex_val = str(hex(int(cipher_suite)))
        hex_val = hex_val[2:].zfill(4)
        cs = cs+hex_val
    
    output = version + "(" + cs + ")"
    return output
    
# handle grease: Joy normalizes all GREASE to 0a0a
# added 64250 (one of the extension values that appear in our db) = fafa?
grease_list = ["2570", "6650", "10794", "14938", "19082", "23226", "27370", "31514", "35658", "39802", "43946", "48090", "52234", "56378", "60522", "64666", "64250"]
apln_layer_prot_nego_dict = {
    "http/0.9": "687474702f302e39",
    "http/1.0": "687474702f312e30",
    "http/1.1": "687474702f312e31",
    "spdy/1": "737064792f31",
    "spdy/2": "737064792f32",
    "spdy/3": "737064792f33",
    "spdy/3.1": "737064792f332e3106",
    "stun.turn": "7374756E2E7475726E",
    "stun.nat-discovery": "7374756E2E6E61742D646973636F76657279",
    "h2": "6832",
    "h2-14": "0568322d313408",
    "h2-15": "0568322d3135",
    "h2-16": "0568322d3136",
    "h2-fb": "0568322d666208",
    "apns-security-v3": "1061706e732d73656375726974792d7633",
    # apns - omit 10?
    "h2c": "683263",
    "webrtc": "776562727463",
    "grpc-exp": "677270632d65787002",
    "c-webrtc": "6364776562727463",
    "ftp": "667470",
    "imap": "696d6170",
    "pop3": "706f7033",
    "managesieve": "6d616e6167657369657665",
    "coap": "636f6170",
    "xmpp-client": "786d70702d636c69656e74",
    "xmpp-server": "786d70702d736572766572",
    "acme-tls/1": "61636d652d746c732f31",
    "mqtt": "6d717474",
    "dot": "646F74",
    "ntske/1": "6e74736b652f31",
    "sunrpc": "73756e727063",
    "h3": "6833",
    "smb": "736d62",
    "irc": "697263",
    "nntp": "6e6e7470",
    "nnsp": "6e6e7370",
    "doq": "646f71",
    "sip/2": "7369702f32",
    "tds/8.0": "7464732f382e30"
}

# third parenthesis (extensions)
def gen_extensions(fp):

    list = fp["handshake"]["client_hello"]["extension_list"]
    result=""
    if len(list) == 0:
        return "()"
    
    for ext in list:
        # not in Joy therefore don't include in our fp
        if ext in grease_list:
            name = "0a0a"
            continue
        if ext in ["65284", "64250", "17513", "22", "44"]:
            continue

        name = str(hex(int(ext)))[2:].zfill(4)
        length=""
        data=""

        # extensions with no value, match for presence
        # if ext in ["65281", "0", "35", "21", "51", "23", "5", "17", "49", "41", "34", "28", "42"]:
            # figure out what to do with 5, 17 (status_request(_v2)) - how to ignore values in Joy db - maybe just delete values from joy?
            # what to do with heartbeat(15) - no value in ours, almost always value in joy, appear 20% so important?
            # 34 doesn't appear in Joy at all - include still? yes probably.
            # 28 (record_size_limit) - Joy almost always has value, ours never - delete from Joy?
            
        if ext=="10":
            # supported_groups
            list_length = hex(2*len(fp["handshake"]["client_hello"]["supported_groups"])+2)
            list_length = str(list_length)[2:].zfill(4)
            print("list length hello "+list_length)
            list_length2 = hex(2*len(fp["handshake"]["client_hello"]["supported_groups"]))
            list_length2 = str(list_length2)[2:].zfill(4)
            length = list_length + list_length2

            # rule: 000a(000x-actual number+2)(000x-actual number)(000x-group)...
            for sg in fp["handshake"]["client_hello"]["supported_groups"]:
                if sg in grease_list:
                    data=data+"0a0a"
                else:
                    data = data+str(hex(int(sg)))[2:].zfill(4)
        elif ext=="11":
            # ec_point_formats
            list_length1 = hex(1+len(fp["handshake"]["client_hello"]["ec_point_formats"]))
            list_length1 = str(list_length1)[2:].zfill(4)
            list_length2 = hex(len(fp["handshake"]["client_hello"]["ec_point_formats"]))
            list_length2 = str(list_length2)[2:].zfill(2)
            length=list_length1+list_length2

            # rule: 000b(000x-actual number+1)(0x-actual number)(0x-group)...
            for ec in fp["handshake"]["client_hello"]["ec_point_formats"]:
                if ec in grease_list:
                    data=data+"0a0a"
                data=data+str(hex(int(ec)))[2:].zfill(2)
            
        elif ext=="13":
            # signature_algorithms
            list_length=hex(2*len(fp["handshake"]["client_hello"]["signature_algs"])+2)
            list_length=str(list_length)[2:].zfill(4)
            list_length2=hex(2*len(fp["handshake"]["client_hello"]["signature_algs"]))
            list_length2=str(list_length2)[2:].zfill(4)
            length=list_length+list_length2

            # length rule: 000d(000x-actual number+2)(000x-actual number)(00xx-alg)...
            for sa in fp["handshake"]["client_hello"]["signature_algs"]:
                if sa in grease_list:
                    data=data+"0a0a"
                data=data+str(hex(int(sa)))[2:].zfill(4)

        elif ext=="43": # supported_versions
            list_length=hex(2*len(fp["handshake"]["client_hello"]["supported_versions"])+1)
            list_length=str(list_length)[2:].zfill(4)
            list_length2=hex(2*len(fp["handshake"]["client_hello"]["supported_versions"]))
            list_length2=str(list_length2)[2:].zfill(2)
            length=list_length+list_length2
            
            # length rule: 002b(000x-actual number+1)(0x-actual number)(00xx-ver)...
            for sv in fp["handshake"]["client_hello"]["supported_versions"]:
                if sv in grease_list:
                    data=data+"0a0a"
                data=data+str(hex(int(sv)))[2:].zfill(4)
        
        # elif ext=="16": # alpn_protocol_alg_nego, what is 02?
        #     # rule: 0010(000e-total number of pairs 0x in protocol encoding+2)(000x-total number of pairs 0x in protocol encoding+1)(02)(0x-number of protocols)(0x-protocol)...            
        #     for item in fp["handshake"]["client_hello"]["alpn_protocols"]:
        #         data = data + apln_layer_prot_nego_dict[item]
        #     length1 = str(hex(int(len(data)/2+3))).zfill(4)
        #     length2 = str(hex(int(len(data)/2+1))).zfill(4)
        #     length=length1+length2+"02"
        #     # sometimes it's 08 and idk why
        #     continue
        
        result = result + "(" + name + length + data + ")"
    return result

def reader(fp_file):
    obj = {}
    with open(fp_file, 'r') as infile:
        with open("/Users/student/Downloads/Joy/joy_fp_created.json", 'w') as outfile:
            for line in infile:
            # Parse the JSON object
                data = json.loads(line)
                result = first_two(data)+gen_extensions(data)
                obj["our_fp"] = data
                obj["joy_fp"] = result
                outfile.write(json.dumps(obj) + '\n')

input_path="/Users/student/Downloads/hs_test.json"

reader(input_path)