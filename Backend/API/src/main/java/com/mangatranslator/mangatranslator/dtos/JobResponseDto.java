package com.mangatranslator.mangatranslator.dtos;

import jakarta.persistence.Column;
import jakarta.persistence.Id;
import lombok.*;

import java.time.LocalDateTime;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
@Data
public class JobResponseDto {
    private String id;

    private String documentId;

    private String sourceLang;

    private String targetLang;

    private String status;

    private Integer progress;

    private String outputPath;

    private String errorMessage;

    private LocalDateTime createdAt;
}
