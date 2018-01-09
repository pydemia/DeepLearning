kernel, recurrent_kernel, bias = model.get_weights()
print(kernel, recurrent_kernel, bias, sep='\n')

return_sequence = False
return_state = True

"""
activation = np.tanh
recurrent_activation = sigmoid
"""

kernel_z = kernel[:,          : units * 1]
kernel_r = kernel[:, units * 1: units * 2]
kernel_h = kernel[:, units * 2:]

recurrent_kernel_z = recurrent_kernel[:,          : units * 1]
recurrent_kernel_r = recurrent_kernel[:, units * 1: units * 2]
recurrent_kernel_h = recurrent_kernel[:, units * 2:]

bias_z = bias[         : units * 1]
bias_r = bias[units    : units * 2]
bias_h = bias[units * 2:]

sequence = []
for _ in range(len(data_X[0])):

    inputs = data_X[0][_]
    inputs = inputs.astype(np.float32)
    print('inputs')
    print(inputs.shape)
    pprint(inputs)

    print('x -----------')
    print('dropout(input) time')
    x_z = np.dot(inputs, kernel_z) + bias_z
    x_r = np.dot(inputs, kernel_r) + bias_r
    x_h = np.dot(inputs, kernel_h) + bias_h

    if _ == 0:
        initial_state = [np.zeros(units, dtype=np.float32)]
        states = initial_state
    else:
        states = [output]
    h_tm1 = prev_output = states[0]
    print('prev --------')
    print(prev_output.shape)
    pprint(prev_output)
    #print('\nprev:', prev_output)

    print('gate --------')
    print('recurrent dropout(h_tm1) time')
    z = sigmoid(x_z + np.dot(h_tm1, recurrent_kernel_z))
    r = sigmoid(x_r + np.dot(h_tm1, recurrent_kernel_r))
    hh = np.tanh(x_h + np.dot(h_tm1, recurrent_kernel_h))

    h = z * h_tm1 + (1 - z) * hh
    h = h.astype(np.float32)
    print(h.shape)
    pprint(h)
    #print('\nh:', h)

    print('a -----------')
    a = np.dot(prev_output, recurrent_kernel)
    a = a.astype(np.float32)
    print(a.shape)
    pprint(a)

    print('output ------')
    output = h
    output = output.astype(np.float32)
    print(output.shape)
    pprint(output)


    print('\nstep (%s)' % _ + '='*37)
    print('h\t\t:', h)
    print('prev\t\t:', prev_output)
    print('output(%s)\t: %s' % (_, output))
    print('='*45 + '\n')

    sequence.append(output)

    if return_sequence:
        result = sequence
    else:
        result = sequence[-1]

    if return_state:
        state = output
    else:
        state = []


result = np.expand_dims(np.stack(result), axis=0)
state = np.expand_dims(np.array(state), axis=0)

print('\nResult: [Output, State]')
print('\n=== Numpy ===')
print([result, state])
print('\n=== Keras ===')
print(model.predict(data_X))
