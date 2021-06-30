from ecs_control import register_task, run_task, wait_for_completion
import boto3
import msgpack
import msgpack_numpy as m
from os import path

def run_subjects(subjlist, do_wait_for_completion=True, output_bucket='neurana-imaging', output_prefix='hcp-test'):   
    response=[]
    for subj in subjlist:
        response.append(run_task(client, command = ['python3','/app/roi_extract_one_subject.py', "--subject", subj,  "--output_bucket",output_bucket, "--output_prefix",output_prefix ]))
    
    if wait_for_completion:
        wait_for_completion(client, response)
    
    return response
    
def get_results(subjlist, output_bucket=None, output_prefix=None):
    session = boto3.session.Session()
    client = session.client('s3')
    allresults={}
    for sub in subjlist:
        outfn=f'sub-{sub}_roi.msgpack-numpy'
        localfn=path.join('/tmp', outfn)
        client.download_file(output_bucket, path.join(output_prefix, outfn),localfn)
        with open(localfn, "rb") as data_file:
            byte_data = data_file.read()
        x_rec = msgpack.unpackb(byte_data, object_hook=m.decode)
        allresults[sub] = x_rec

    return allresults

if __name__=='__main__':
    output_bucket='neurana-imaging'
    output_prefix='hcp_test'
    session = boto3.session.Session()
    client = session.client('ecs', region_name='eu-west-1')
    #response = register_task(client)
    subjlist = ['178950','189450']
    #subjlist = ['178950','189450','199453','209228','220721','298455','356948','419239','499566','561444','618952','680452','757764','841349','908860', '103818','113922','121618','130619','137229','151829','158035','171633','179346','190031','200008','210112','221319','299154','361234', '424939','500222','570243','622236','687163','769064','845458','911849','104416','114217','122317','130720','137532','151930','159744', '172029','180230','191235','200614','211316','228434','300618','361941','432332','513130','571144','623844','692964','773257','857263', '926862','105014','114419','122822','130821','137633','152427','160123','172938','180432','192035','200917','211417','239944','303119', '365343','436239','513736','579665','638049','702133','774663','865363','930449','106521','114823','123521','130922','137936','152831', '160729','173334','180533','192136','201111','211619','249947','305830','366042','436845','516742','580650','645450','715041','782561', '871762','942658','106824','117021','123925','131823','138332','153025','162026','173536','180735','192439','201414','211821','251833', '310621','371843','445543','519950','580751','647858','720337','800941','871964','955465','107018','117122','125222','132017','138837', '153227','162329','173637','180937','193239','201818','211922','257542','314225','378857','454140','523032','585862','654350','725751', '803240','872562','959574','107422','117324','125424','133827','142828','153631','164030','173940','182739','194140','202719','212015', '257845','316633','381543','459453','525541','586460','654754','727553','812746','873968','966975']
    #response = run_subjects(subjlist, output_bucket=output_bucket, output_prefix=output_prefix)
    allresults=get_results(subjlist, output_bucket=output_bucket, output_prefix=output_prefix )
    print(allresults)