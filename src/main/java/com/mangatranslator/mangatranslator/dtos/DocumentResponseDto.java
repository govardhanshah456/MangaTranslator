package com.mangatranslator.mangatranslator.dtos;

import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
public class DocumentResponseDto {
    private String documentId;
    private String jobId;
    private String status;
}
